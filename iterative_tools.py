from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

# Needs a recent LangGraph for interrupt/Command:
#   pip install -U langgraph langchain-groq langchain-tavily


# ============================================================
# 1. LOAD API KEYS
# ============================================================

load_dotenv()


# ============================================================
# 2. LIMITS (per run only; set to ~80% of your real free-tier limits)
# ============================================================

MAX_ATTEMPTS = 3            # max drafts per run
MAX_TOKENS_PER_RUN = 8000   # Groq tokens allowed in one run
RESERVE_PER_CALL = 1500     # safety margin: rough worst-case size of ONE call
MAX_SEARCHES_PER_RUN = 1    # Tavily calls allowed in one run

# Per-run counters (reset at the start of every run, see bottom)
RUN = {"tokens": 0, "searches": 0}


class BudgetExceeded(Exception):
    """Raised when one more Groq call could cross the token limit."""


def call_llm(llm, messages):
    """
    ONE place for every Groq call:
      1. check the budget BEFORE calling
      2. call the model
      3. record the tokens it used AFTER
    """
    if RUN["tokens"] + RESERVE_PER_CALL > MAX_TOKENS_PER_RUN:
        raise BudgetExceeded("Per-run token limit reached.")

    response = llm.invoke(messages)

    used = (getattr(response, "usage_metadata", None) or {}).get("total_tokens", 0)
    RUN["tokens"] += used
    print(f"   tokens: +{used} | run {RUN['tokens']}/{MAX_TOKENS_PER_RUN}")
    return response


# ============================================================
# 3. SEARCH TOOL
# ============================================================

# basic depth = cheaper; 2 results; no extra answer/raw content
_tavily = TavilySearch(
    max_results=2,
    search_depth="basic",
    include_answer=False,
    include_raw_content=False,
)


# @tool lets the LLM see this function and decide to call it.
@tool
def web_search(query: str) -> str:
    """Search the web for current facts, stats or trends.
    Use only if the topic really needs fresh information."""

    RUN["searches"] += 1
    print(f"   Tavily search: {query[:150]}")

    try:
        result = _tavily.invoke({"query": query[:150]})
    except Exception as e:
        return f"Search failed ({e})."

    # Keep only title + short snippet (drops urls/scores = fewer tokens)
    items = result.get("results", []) if isinstance(result, dict) else []
    return "\n".join(
        f"- {r.get('title', '')}: {r.get('content', '')[:400]}" for r in items
    ) or "No results found."


# ============================================================
# 4. LLM
# ============================================================
# max_retries=0     -> no hidden retries eating quota
# max_tokens        -> caps output (reasoning tokens count too)
# reasoning_effort  -> less hidden "thinking" = fewer tokens

writer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    max_tokens=900,
    max_retries=0,
    reasoning_effort="low",
)

# Same model, but it is ALLOWED to ask for web_search.
writer_with_tools = writer_llm.bind_tools([web_search])


# ============================================================
# 5. STATE
# ============================================================

class State(TypedDict):
    topic: str
    research: str          # search results, kept so rewrites reuse them
    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


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


# ============================================================
# 6. WRITER NODE
# ============================================================

def writer_node(state: State) -> dict:
    """Writes (or rewrites) the LinkedIn post. May search the web once."""
    attempt = state["attempt"] + 1
    topic = state["topic"]
    research = state["research"]

    print(f"\n[Attempt {attempt}] Writer is drafting the post...")

    if attempt == 1:
        user_message = f"Write a LinkedIn post on this topic: {topic}"
    else:
        # Include the previous draft, or the writer can't know what to fix.
        user_message = (
            f"Your previous draft on '{topic}' was rejected.\n\n"
            f"Previous draft:\n{state['draft']}\n\n"
            f"Human feedback:\n{state['review_feedback']}\n\n"
            "Write a NEW improved LinkedIn post that fixes every issue mentioned."
        )

    # Reuse old research on rewrites (no extra Tavily credit).
    if research:
        user_message += f"\n\nResearch (use only these facts):\n{research}"

    messages = [("system", WRITER_SYSTEM_PROMPT), ("human", user_message)]

    # Searching is allowed only if we have no research yet AND budget is left.
    can_search = (not research) and RUN["searches"] < MAX_SEARCHES_PER_RUN
    response = call_llm(writer_with_tools if can_search else writer_llm, messages)

    # Did the writer ask for a search? Run it ourselves with a plain `if`.
    # (No ToolNode needed: one search, then one final writing call.)
    if response.tool_calls:
        research = web_search.invoke(response.tool_calls[0]["args"])
        messages.append(("human", f"Research (use only these facts):\n{research}"))
        response = call_llm(writer_llm, messages)   # no tools on this call

    draft = response.content.strip()

    return {"draft": draft, "research": research, "attempt": attempt}


# ============================================================
# 7. HUMAN REVIEW NODE (the HITL part)
# ============================================================

def human_review_node(state: State) -> dict:
    """Pauses the graph and waits for the human to approve or give feedback."""

    # interrupt() PAUSES here and saves state (MemorySaver).
    # The dict we pass is what the human sees. When we resume with
    # Command(resume=...), interrupt() RETURNS that resume value.
    #
    # NOTE: on resume, this whole node re-runs from the top, so keep
    # prints / API calls / file writes OUT of the code above interrupt().
    human_response = interrupt({
        "draft": state["draft"],
        "attempt": state["attempt"],
        "instruction": "Type 'approved' to accept, or type your feedback to request a rewrite.",
    })

    response = human_response.strip()

    if response.lower() in ["approved", "approve", "yes", "ok", "good"]:
        return {
            "is_approved": True,
            "review_feedback": "Approved by human."
        }
    else:
        return {
            "is_approved": False,
            "review_feedback": response[:500]   # feedback goes into the next prompt, so cap it
        }


# ============================================================
# 8. ROUTER (the loop: human_review -> writer -> human_review)
# ============================================================

def should_stop_looping(state: State):
    if state["is_approved"]:
        print("\n[Post approved by human. Ending workflow.]")
        return END
    if state["attempt"] >= MAX_ATTEMPTS:
        print(f"\n[Reached max {MAX_ATTEMPTS} attempts. Ending with last draft.]")
        return END
    print(f"\n[Rejected. Looping back to writer for attempt {state['attempt'] + 1}...]")
    return "writer"


# ============================================================
# 9. BUILD THE GRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node("writer", writer_node)
graph.add_node("human_review", human_review_node)

graph.add_edge(START, "writer")
graph.add_edge("writer", "human_review")

graph.add_conditional_edges(
    "human_review",
    should_stop_looping,
    {
        "writer": "writer",
        END: END,
    },
)

# MemorySaver keeps graph state in RAM so interrupt() can pause and resume.
app = graph.compile(checkpointer=MemorySaver())


# ============================================================
# 10. RUN THE APPLICATION
# ============================================================

print("=" * 55)
print("Welcome to the LinkedIn Post Generator (HITL Edition)")
print("=" * 55)
print("\nThis tool will draft a LinkedIn post for you, show it to")
print("YOU for review, and rewrite based on your feedback.")
print("=" * 55)

topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

if not topic:
    print("\nNo topic given. Exiting.")
elif len(topic) > 100:
    print("\nTopic too long (max 100 characters).")
else:
    print("\nStarting generation...\n")

    # thread_id = the "save slot" MemorySaver uses for this run.
    # recursion_limit = safety net against runaway loops (= runaway tokens).
    config = {
        "configurable": {"thread_id": "linkedin_session_1"},
        "recursion_limit": MAX_ATTEMPTS * 2 + 4,
    }

    initial_state = {
        "topic": topic,
        "research": "",
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    try:
        result = app.invoke(initial_state, config=config)

        # While the graph is paused at interrupt(), ask the human and resume.
        while "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0].value

            print("\n" + "=" * 55)
            print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
            print("=" * 55)
            print(interrupt_data["draft"])
            print("=" * 55)
            print(f"\n{interrupt_data['instruction']}")

            # Empty feedback would waste a rewrite, so ask again.
            human_input = ""
            while not human_input:
                human_input = input("\nYour response: ").strip()

            result = app.invoke(Command(resume=human_input), config=config)

    except BudgetExceeded as e:
        print(f"\nSTOPPED: {e}")
    except Exception as e:
        print(f"\nRun failed (maybe a rate limit): {e}")

    # Read the last saved state (works even after a budget stop).
    final = app.get_state(config).values

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final.get("draft", "(no draft produced)"))
    print("=" * 55)
    print(f"Total attempts: {final.get('attempt', 0)}")
    print(f"Approved by human: {final.get('is_approved', False)}")
    print(f"Tokens this run: {RUN['tokens']} | Searches: {RUN['searches']}")


# ============================================================
# WORKFLOW IN SHORT
# ============================================================
#
# START
#   -> WRITER: drafts the post. If it needs fresh facts it asks for
#      web_search; we run Tavily ONCE and the writer finishes the post.
#   -> HUMAN_REVIEW: graph PAUSES (interrupt). You type:
#        - "approved"      -> END
#        - your feedback   -> back to WRITER (rewrite), up to MAX_ATTEMPTS
#   -> END: final post is printed
#
# Safety nets: call_llm() checks the token budget before every Groq call,
# max 1 search per run, capped topic/feedback length, max_tokens,
# max_retries=0, recursion_limit.
# MemorySaver + thread_id is what lets the graph pause and resume.