import os
from dotenv import load_dotenv
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from langchain_community.tools import tool
from langchain_core.messages import AnyMessage
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.semanticscholar import SemanticScholarQueryRun
from langchain_community.utilities.semanticscholar import SemanticScholarAPIWrapper
from langchain_tavily import TavilySearch
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage

# Load environment
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# --- LLM setup
llm = init_chat_model("gpt-4.1", temperature=0.7)

# --- Tools
arxiv_tool = Tool(
    name="arxiv_search",
    description="Searches Arxiv.org for academic papers. Input a topic or paper ID.",
    func=ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=5)).run,
)
semanticscholar_tool = Tool(
    name="semantic_scholar_search",
    description="Finds research papers and citation info using Semantic Scholar.",
    func=SemanticScholarQueryRun(api_wrapper=SemanticScholarAPIWrapper()).run,
)
wikipedia_tool = Tool(
    name="wikipedia_search",
    description="Looks up scientific info on Wikipedia for background knowledge.",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2)).run,
)
tavily_tool = Tool(
    name="tavily_search",
    description="Performs a real-time web search for the latest scientific news and blog summaries.",
    func=TavilySearch().run,
)
pubmed_tool = PubmedQueryRun()
pubmed_search = Tool(
    name="pubmed_search",
    description="A tool for searching the PubMed database for biomedical literature and extracting information.",
    func=pubmed_tool.run,
)
python_tool = PythonREPLTool()
tools = [arxiv_tool, semanticscholar_tool, wikipedia_tool, tavily_tool, pubmed_search, python_tool]

# --- System Prompt
react_prompt = """
You are a scientific research assistant capable of searching for papers, comparing findings, and synthesizing up-to-date answers.

Always reason step by step before acting:
- For each user question, break down what information is needed.
- Select and invoke the most relevant tool for each subtask.
- For each tool, explain your reasoning and why you chose that tool.
- Summarize each tool output and decide if another step is needed.
- Finally, synthesize all findings into a concise answer, citing sources.

Say “I don’t know” if you can’t find enough relevant information.

**Instructions:**
- Favor the latest publications where relevant.
- Use other tools if your initial source is insufficient.
---
Example interaction:
User: What are the major differences between GPT-3 and GPT-4 according to recent research?
Step 1: I will search arXiv for recent papers comparing GPT-3 and GPT-4.
[arxiv_search: "GPT-3 GPT-4 comparison"]
Summary...
Now begin reasoning step by step for each user question.
"""

# --- State
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# --- Graph Node: Reasoning
def reasoning_node(state: State):
    llm_with_tools = llm.bind_tools(tools)
    messages = [{"role": "system", "content": react_prompt}] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": state["messages"] + [response]}

# --- Graph Node: ToolNode
tool_node = ToolNode(tools=tools)

# --- Graph Node: Grader (Self-reflection)
def grader_node(state: State):
    # Append a grader self-reflection message to state
    last_content = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else ""
    grader_prompt = (
        "Did the answer below fully and clearly cover all aspects of the user's research question using the best sources? "
        "If everything was addressed, state YES. If anything is missing or should be improved/expanded, state what to do next, briefly.\n"
        f"====\n{last_content}\n===="
    )
    review = llm.invoke([{"role": "system", "content": grader_prompt}])
    # Always return a dict with messages (LangGraph requirement)
    return {"messages": state["messages"] + [review]}

# --- Grader router (for conditional edge)
def grader_router(state: State):
    # Only look at the last grader review message for decision
    review_msg = getattr(state["messages"][-1], "content", "").lower()
    # Add more flexible stopping - accept 'yes', 'fully addressed', or similar
    stop_keywords = ["yes", "fully", "addressed", "fully answered", "covers all"]
    # You can tune the below as you see patterns in the grader LLM's output
    if any(kw in review_msg for kw in stop_keywords):
        return END
    if "i don't know" in review_msg:
        return END
    # Safety: stop if we're really looping a lot (prevents recursion error)
    if len(state["messages"]) > 20:
        return END
    # If you want the agent to attempt a new reasoning step, return "reason"
    return "reason"

# --- Should_continue router (for conditional edge)
def should_continue(state: State):
    last_message = state["messages"][-1]
    content = getattr(last_message, "content", "").lower()
    # Heuristic: If the agent completes with 'final answer:' or 'final summary:' or covers all in one step
    if "final answer:" in content or "final summary:" in content:
        return "grader"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "action"
    if len(state["messages"]) > 20:
        return "grader"
    return "grader"

checkpointer = InMemorySaver()

builder = StateGraph(State)
builder.add_node("reason", reasoning_node)
builder.add_node("action", tool_node)
builder.add_node("grader", grader_node)
builder.set_entry_point("reason")
builder.add_conditional_edges(
    "reason", should_continue, {"action": "action", "grader": "grader"}
)
builder.add_edge("action", "reason")
builder.add_conditional_edges("grader", grader_router, {"reason": "reason", END: END})
research_agent = builder.compile(checkpointer=checkpointer)