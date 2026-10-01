import html
import os
import re
import threading
import time
import uuid
from datetime import date
from typing import TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# Must be the FIRST Streamlit call.
st.set_page_config(page_title="PostPilot", page_icon="✈️", layout="centered")

# Local dev: keys from .env. Deployed: keys from .streamlit/secrets.toml
load_dotenv()
try:
    for _k in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass  # no secrets file locally, that's fine


# ============================================================
# 1. SETTINGS  (tune these to your real free-tier limits)
# ============================================================

# --- per visitor ---
MAX_RUNS_PER_SESSION = 3        # new posts per visitor session
COOLDOWN_SECONDS = 30           # wait time between runs
MAX_TOPIC_CHARS = 100
MAX_FEEDBACK_CHARS = 300
MAX_ATTEMPTS = 2                # drafts per post (1 draft + 1 rewrite)

# --- per run ---
MAX_TOKENS_PER_RUN = 6000
RESERVE_PER_CALL = 1500         # safety margin: rough worst-case size of ONE call

# --- whole demo, all visitors together (resets daily, and on app restart) ---
MAX_RUNS_PER_DAY = 20
DAILY_TOKEN_BUDGET = 60000
MAX_SEARCHES_PER_DAY = 15

PUBLIC_MODEL = "openai/gpt-oss-20b"     # cheaper model for the public demo
BYOK_MODEL = "openai/gpt-oss-120b"      # stronger model when a visitor brings their own key

EMPTY_DRAFT_MSG = "The model returned an empty draft. Please try again."

EXAMPLE_TOPICS = [
    "Why human-in-the-loop matters in AI",
    "Career tips for data analysts",
    "AI agents for small businesses",
]

SAMPLE_POST = """AI shouldn't publish for you. It should draft for you.

I tested a simple idea: let an AI write the first version of a post, then pause and wait for a human decision.

The result? Better drafts, fewer surprises, and zero "wait, I never approved that."

Human-in-the-loop isn't a slowdown. It's a safety net that keeps your voice in the final result.

The AI handles the blank page. You handle the judgment.

Where in your workflow would a pause for human approval save you the most trouble?"""

NODE_LABELS = {
    "writer": "Writer drafted your post",
    "human_review": "Your decision was recorded",
}


# ============================================================
# 2. SHARED RESOURCES
# ============================================================
# Streamlit re-runs this whole script on EVERY click, so anything that must
# survive between clicks/visitors lives in st.cache_resource.

@st.cache_resource
def get_checkpointer():
    # MemorySaver must survive reruns, or interrupt() could not resume.
    return MemorySaver()


@st.cache_resource
def get_tavily():
    return TavilySearch(
        max_results=2,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )


@st.cache_resource
def get_draft_cache():
    # topic -> first draft. Same topic again = 0 tokens, 0 searches.
    return {}


@st.cache_resource
def _store():
    # Global counters shared by ALL visitors.
    return {
        "lock": threading.Lock(),
        "data": {"date": "", "runs": 0, "tokens": 0, "searches": 0},
    }


def _usage(store, key=None, add=0):
    """Reads (and optionally bumps) today's global counters. Thread-safe."""
    with store["lock"]:
        d = store["data"]
        if d["date"] != str(date.today()):
            d.update(date=str(date.today()), runs=0, tokens=0, searches=0)
        if key:
            d[key] += add
        return dict(d)


def usage(key=None, add=0):
    """Same thing, for UI code running in the main script thread."""
    return _usage(_store(), key, add)


class BudgetExceeded(Exception):
    """Raised when one more LLM call could cross a token limit."""


# ============================================================
# 3. LANGGRAPH: STATE, NODES, GRAPH
# ============================================================

class State(TypedDict):
    topic: str
    research: str
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int
    tokens_used: int      # per-run counter lives in state, so visitors never mix
    use_search: bool


WRITER_SYSTEM_PROMPT = (
    "You are an expert LinkedIn content writer. Write engaging, professional "
    "LinkedIn posts about the given topic. "
    "Rules: strong hook in the first line, one clear takeaway, easy to skim "
    "with short paragraphs, roughly 150-200 words, end with an engaging "
    "question or CTA, no hashtags. "
    "Use the web_search tool only if fresh facts are really needed. "
    "Never invent stats, dates, prices or specs. "
    "If you receive feedback on a previous draft, address every point carefully. "
    "Output the post only."
)


def make_llm(byok_key):
    kwargs = {"api_key": byok_key} if byok_key else {}
    return ChatGroq(
        model=BYOK_MODEL if byok_key else PUBLIC_MODEL,
        temperature=0.5,
        max_tokens=900,          # output cap (reasoning tokens count too)
        max_retries=0,           # no hidden retries eating quota
        reasoning_effort="low",  # less hidden "thinking" = fewer tokens
        **kwargs,
    )


def human_review_node(state: State) -> dict:
    """Pauses the graph. The dict passed to interrupt() is what the UI receives."""
    # On resume this node re-runs from the top, so no side effects above interrupt().
    human_response = interrupt({
        "draft": state["draft"],
        "attempt": state["attempt"],
    })
    response = human_response.strip()

    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {"is_approved": True, "review_feedback": "Approved by human."}
    return {"is_approved": False, "review_feedback": response[:MAX_FEEDBACK_CHARS]}


def should_stop_looping(state: State):
    if state["is_approved"] or state["attempt"] >= MAX_ATTEMPTS:
        return END
    return "writer"


def route_start(state: State):
    # Cached first draft? Skip the writer and go straight to the human.
    return "human_review" if state["draft"] else "writer"


def build_app(byok_key):
    """
    Builds the graph for ONE run. Resources (counters, Tavily) are grabbed here,
    in the main thread, and used inside the nodes via closures.
    The checkpointer is shared, so a freshly built graph can resume an old thread.
    """
    store = _store()
    tavily = get_tavily()
    byok = bool(byok_key)
    llm = make_llm(byok_key)

    def u(key=None, add=0):
        return _usage(store, key, add)

    @tool
    def web_search(query: str) -> str:
        """Search the web for current facts, stats or trends.
        Use only if the topic really needs fresh information."""
        if u()["searches"] >= MAX_SEARCHES_PER_DAY:
            return "Search quota reached. Write the post without searching."
        u("searches", 1)
        try:
            result = tavily.invoke({"query": query[:150]})
        except Exception:
            return "Search failed. Write the post without searching."
        items = result.get("results", []) if isinstance(result, dict) else []
        return "\n".join(
            f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in items
        ) or "No results found."

    llm_tools = llm.bind_tools([web_search])   # writer that MAY search

    def call_llm(model, messages, run_tokens):
        """Budget check -> call -> count tokens. Every Groq call goes through here."""
        if run_tokens + RESERVE_PER_CALL > MAX_TOKENS_PER_RUN:
            raise BudgetExceeded("This run reached its token limit.")
        if not byok and u()["tokens"] + RESERVE_PER_CALL > DAILY_TOKEN_BUDGET:
            raise BudgetExceeded("The demo's daily token budget is used up.")

        response = model.invoke(messages)
        used = (getattr(response, "usage_metadata", None) or {}).get("total_tokens", 0)
        if not byok:                # BYOK tokens are the visitor's, not ours
            u("tokens", used)
        return response, used

    def writer_node(state: State) -> dict:
        attempt = state["attempt"] + 1
        topic = state["topic"]
        research = state["research"]
        tokens = state["tokens_used"]

        if attempt == 1:
            user_message = f"Write a LinkedIn post on this topic: {topic}"
        else:
            user_message = (
                f"Your previous draft on '{topic}' was rejected.\n\n"
                f"Previous draft:\n{state['draft']}\n\n"
                f"Human feedback:\n{state['review_feedback']}\n\n"
                "Write a NEW improved LinkedIn post that fixes every issue mentioned."
            )
        if research:
            user_message += f"\n\nResearch (use only these facts):\n{research}"

        messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]

        # Search only if: visitor allows it, we have no research yet, quota left.
        can_search = (
            state["use_search"] and not research and u()["searches"] < MAX_SEARCHES_PER_DAY
        )
        response, used = call_llm(llm_tools if can_search else llm, messages, tokens)
        tokens += used

        if response.tool_calls:   # writer asked for a search: run it once, then finish
            research = web_search.invoke(response.tool_calls[0]["args"])
            messages.append(("human", f"Research (use only these facts):\n{research}"))
            response, used = call_llm(llm, messages, tokens)
            tokens += used

        draft = (response.content or "").strip() or EMPTY_DRAFT_MSG
        return {"draft": draft, "research": research, "attempt": attempt, "tokens_used": tokens}

    graph = StateGraph(State)
    graph.add_node("writer", writer_node)
    graph.add_node("human_review", human_review_node)

    graph.add_conditional_edges(START, route_start, {"writer": "writer", "human_review": "human_review"})
    graph.add_edge("writer", "human_review")
    graph.add_conditional_edges("human_review", should_stop_looping, {"writer": "writer", END: END})

    return graph.compile(checkpointer=get_checkpointer())


# ============================================================
# 4. SESSION STATE  (one per visitor)
# ============================================================

ss = st.session_state
for _key, _val in {
    "phase": "idle",              # idle -> review -> done
    "thread_id": str(uuid.uuid4()),
    "runs_used": 0,
    "last_run_ts": 0.0,
    "payload": None,              # what interrupt() sent us
    "final": None,
    "topic_input": "",
}.items():
    ss.setdefault(_key, _val)


# ============================================================
# 5. STYLE
# ============================================================

st.markdown("""
<style>
.block-container {max-width: 820px; padding-top: 2rem;}
#MainMenu, footer {visibility: hidden;}
.pp-hero h1 {font-size: 2.8rem; font-weight: 800; margin: 0;}
.pp-grad {background: linear-gradient(90deg,#2563eb,#7c3aed);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.pp-tag {font-size: 1.3rem; font-weight: 600; margin: .2rem 0;}
.pp-sub {opacity: .75; margin-bottom: .8rem;}
.pp-badge {display: inline-block; padding: .15rem .65rem; margin: 0 .3rem .4rem 0;
           border-radius: 999px; background: rgba(37,99,235,.12); font-size: .78rem;}
.pp-recruiter {border-left: 4px solid #7c3aed; background: rgba(124,58,237,.08);
               padding: .9rem 1.2rem; border-radius: 8px; margin: 1rem 0 1.4rem 0;}
.pp-draft {white-space: pre-wrap; line-height: 1.65;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 6. HELPERS
# ============================================================

def daily_exhausted() -> bool:
    g = usage()
    return g["runs"] >= MAX_RUNS_PER_DAY or g["tokens"] + RESERVE_PER_CALL > DAILY_TOKEN_BUDGET


def check_limits(topic, byok, cached):
    """Returns an error message, or None if the run may start."""
    if not topic:
        return "Please enter a topic first."
    if len(topic) > MAX_TOPIC_CHARS:
        return f"Topic is too long (max {MAX_TOPIC_CHARS} characters)."
    if byok or cached:          # own key or cached draft = costs us nothing
        return None
    if ss.runs_used >= MAX_RUNS_PER_SESSION:
        return (f"You've used your {MAX_RUNS_PER_SESSION} demo runs. "
                "Paste your own Groq key in the sidebar to keep going.")
    wait = COOLDOWN_SECONDS - (time.time() - ss.last_run_ts)
    if wait > 0:
        return f"Cooldown: please wait {int(wait) + 1}s before the next run."
    if daily_exhausted():
        return "Today's demo quota is used up. Come back tomorrow, or use your own Groq key."
    return None


def plain_text(draft: str) -> str:
    """
    Markdown -> plain text, for the copy box and the download.
    LinkedIn does not render markdown, so **bold** would show as raw asterisks there.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", draft)            # **bold** -> bold
    text = re.sub(r"__(.+?)__", r"\1", text)                  # __bold__ -> bold
    text = re.sub(r"^#{1,6}[ \t]*", "", text, flags=re.M)     # "# Heading" -> "Heading"
    text = re.sub(r"^[ \t]*[-*][ \t]+", "• ", text, flags=re.M)  # "- item" -> "• item"
    return text.strip()


def show_post(draft: str):
    """Shows a post in a card. Markdown is rendered, so **bold** and # headings look right."""
    words = len(draft.split())
    st.markdown(f'<span class="pp-badge">~{words} words</span>', unsafe_allow_html=True)
    with st.container(border=True):
        # "  \n" (two spaces + newline) keeps single line breaks, like the model wrote them.
        # st.markdown without unsafe_allow_html escapes raw HTML, so this is safe.
        st.markdown(draft.replace("\n", "  \n"))


def run_graph(graph_input, byok_key, cache_key=None):
    """
    Runs/resumes the graph with live progress.
    stream_mode="updates" gives one chunk per finished node;
    an "__interrupt__" chunk means the graph paused for the human.
    """
    app = build_app(byok_key)
    config = {
        "configurable": {"thread_id": ss.thread_id},
        "recursion_limit": MAX_ATTEMPTS * 2 + 4,   # safety net against runaway loops
    }
    payload, notice = None, None

    with st.status("PostPilot is working...", expanded=True) as status:
        try:
            for chunk in app.stream(graph_input, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        payload = update[0].value
                        continue
                    st.write(f"✅ {NODE_LABELS.get(node, node)}")

                    # Cache the first public draft so the same topic is free next time.
                    if node == "writer" and cache_key and update.get("attempt") == 1 \
                            and update["draft"] != EMPTY_DRAFT_MSG:
                        cache = get_draft_cache()
                        if len(cache) < 50:
                            cache[cache_key] = {"draft": update["draft"], "research": update["research"]}
            status.update(label="Ready", state="complete", expanded=False)
        except BudgetExceeded as e:
            notice = f"{e} Showing your latest draft, if there is one."
            status.update(label="Stopped", state="error")
        except Exception as e:
            print(f"Run failed: {e}")   # full error only in the server log
            notice = "Something went wrong (likely a rate limit). Please try again in a minute."
            status.update(label="Something went wrong", state="error")

    values = app.get_state(config).values
    if not payload:   # run finished: stats go to the server log only, visitors don't see them
        print(f"[usage] drafts={values.get('attempt', 0)} tokens={values.get('tokens_used', 0)} "
              f"research={'yes' if values.get('research') else 'no'}")
    ss.notice = notice
    if payload:
        ss.phase, ss.payload = "review", payload
    else:
        ss.final = values
        ss.phase = "done" if values.get("draft") else "idle"
    st.rerun()


def start_run(topic, use_search, byok_key):
    byok = bool(byok_key)
    cache_key = None if byok else (topic.lower(), use_search)
    cached = get_draft_cache().get(cache_key) if cache_key else None

    ss.thread_id = str(uuid.uuid4())          # fresh save slot per post
    if cached:
        st.toast("⚡ Loaded a cached draft (0 tokens used)")
    else:
        ss.runs_used += 1
        ss.last_run_ts = time.time()
        if not byok:
            usage("runs", 1)

    initial = {
        "topic": topic,
        "research": cached["research"] if cached else "",
        "draft": cached["draft"] if cached else "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 1 if cached else 0,
        "tokens_used": 0,
        "use_search": use_search,
    }
    run_graph(initial, byok_key, None if cached else cache_key)


def reset():
    ss.phase, ss.payload, ss.final = "idle", None, None
    st.rerun()


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ✈️ PostPilot")
    st.caption("Your demo session")

    byok_key = st.text_input(
        "Your Groq API key (optional)", type="password", key="byok_key",
        help="Unlimited runs with your own key. Used only during this session, never stored.",
    ).strip() or None
    byok = bool(byok_key)

    if byok:
        st.success("Using your key: no demo limits.")
    else:
        left = max(MAX_RUNS_PER_SESSION - ss.runs_used, 0)
        st.progress(left / MAX_RUNS_PER_SESSION)
        st.caption(f"{left} of {MAX_RUNS_PER_SESSION} runs left in your session")
        day_left = max(MAX_RUNS_PER_DAY - usage()["runs"], 0)
        st.caption(f"Demo runs left today (everyone): {day_left}")

    use_search = st.toggle("🔎 Live web search", value=True,
                           help="Lets the writer look up fresh facts (max 1 search per post).")

    st.divider()
    st.caption("Built with LangGraph · Groq · Tavily · Streamlit")


# ============================================================
# 8. MAIN PAGE
# ============================================================

st.markdown("""
<div class="pp-hero">
  <h1>✈️ <span class="pp-grad">PostPilot</span></h1>
  <div class="pp-tag">AI drafts. You decide.</div>
  <div class="pp-sub">An AI writer that researches the web, drafts your LinkedIn post,
  and waits for your approval before anything is final.</div>
  <span class="pp-badge">LangGraph</span><span class="pp-badge">Groq</span>
  <span class="pp-badge">Tavily</span><span class="pp-badge">Streamlit</span>
  <span class="pp-badge">Human-in-the-loop</span>
</div>
<div class="pp-recruiter">
  👋 <b>Curious how it works? Test it in 60 seconds.</b><br>
  1) Pick a topic &rarr; 2) Review the AI's draft &rarr; 3) Give feedback or approve.<br>
  Features human-in-the-loop approval, agentic web search, and a token budget per user.
  <i>Free-tier demo, so each visitor gets a few runs.</i>
</div>
""", unsafe_allow_html=True)

notice = ss.pop("notice", None)
if notice:
    st.warning(notice)


# ---------------- phase: idle ----------------
if ss.phase == "idle":
    if not byok and daily_exhausted():
        st.info("Today's demo quota is used up. Here's a sample of what PostPilot writes, "
                "or paste your own Groq key in the sidebar to run it live.")
        with st.expander("See a sample post"):
            show_post(SAMPLE_POST)

    st.text_input("What should your post be about?", key="topic_input",
                  max_chars=MAX_TOPIC_CHARS, placeholder="e.g. Why human-in-the-loop matters in AI")

    st.caption("Or try an example:")
    cols = st.columns(len(EXAMPLE_TOPICS))
    for col, example in zip(cols, EXAMPLE_TOPICS):
        col.button(example, key=f"ex_{example}", on_click=lambda t=example: ss.update(topic_input=t))

    if st.button("✨ Generate draft", type="primary"):
        topic = ss.topic_input.strip()
        cache_key = None if byok else (topic.lower(), use_search)
        cached = bool(cache_key and get_draft_cache().get(cache_key))
        error = check_limits(topic, byok, cached)
        if error:
            st.warning(error)
        else:
            start_run(topic, use_search, byok_key)


# ---------------- phase: review (graph is paused at interrupt) ----------------
elif ss.phase == "review":
    payload = ss.payload
    attempt = payload["attempt"]
    last_attempt = attempt >= MAX_ATTEMPTS

    st.markdown(f'<span class="pp-badge">Attempt {attempt} of {MAX_ATTEMPTS}</span>',
                unsafe_allow_html=True)
    show_post(payload["draft"])

    if last_attempt:
        st.info("That's the last draft for this demo. Approve it to finish.")
        feedback = ""
    else:
        feedback = st.text_area("Want changes? Tell PostPilot what to fix:",
                                key=f"fb_{ss.thread_id}_{attempt}", max_chars=MAX_FEEDBACK_CHARS,
                                placeholder="e.g. Make the hook punchier and shorter")

    c1, c2 = st.columns(2)
    approve = c1.button("✅ Approve & finish", type="primary")
    rewrite = c2.button("🔁 Rewrite with my feedback", disabled=last_attempt)

    if approve:
        run_graph(Command(resume="approved"), byok_key)      # interrupt() returns "approved"
    if rewrite:
        if not feedback.strip():
            st.warning("Write some feedback first, so the rewrite has something to fix.")
        else:
            run_graph(Command(resume=feedback.strip()[:MAX_FEEDBACK_CHARS]), byok_key)


# ---------------- phase: done ----------------
elif ss.phase == "done":
    final = ss.final
    st.success("Your post is ready." if final.get("is_approved") else "Here is your latest draft.")
    show_post(final["draft"])

    st.download_button("⬇️ Download as .txt", plain_text(final["draft"]), file_name="postpilot_post.txt")
    with st.expander("Copy-friendly text"):
        st.code(plain_text(final["draft"]), language=None)

    if st.button("✨ New post"):
        reset()


# ============================================================
# WORKFLOW IN SHORT
# ============================================================
# 1. Visitor picks a topic -> check_limits() (session runs, cooldown, daily quota).
# 2. Cached topic? Jump straight to human review (route_start), 0 tokens.
# 3. WRITER drafts (may call web_search once) -> HUMAN_REVIEW pauses via interrupt().
# 4. The UI shows the draft. Visitor approves or sends feedback.
#    Command(resume=...) resumes the SAME thread (MemorySaver + thread_id).
# 5. Feedback -> writer rewrites (max MAX_ATTEMPTS). Approve -> END -> final card.
# Token safety: per-run cap in state, daily budget, daily search cap, cache,
# smaller public model, BYOK for unlimited use, recursion_limit.