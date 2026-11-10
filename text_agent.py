from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated
import operator
import re
from collections import Counter

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import (
    AnyMessage, SystemMessage, ToolMessage, HumanMessage,
)
from langgraph.graph import StateGraph, START, END

load_dotenv()

model = init_chat_model("google_genai:gemini-3.5-flash-lite", timeout=30)



STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "are", "was", "were", "has",
    "have", "had", "from", "not", "but", "can", "will", "its", "into", "than",
    "then", "them", "they", "their", "which", "also", "only", "each", "when",
}
@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""
    return len(text.split())


@tool
def sentence_count(text: str) -> int:
    """Count the number of sentences in a piece of text."""
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


@tool
def top_keywords(text: str, n: int = 5) -> list[str]:
    """Return the n most frequent meaningful words in the text, ignoring common stopwords."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(n)]


tools = [word_count, sentence_count, top_keywords]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def llm_call(state: AgentState):
    """The LLM decides whether to call a tool or answer."""
    system = SystemMessage(
        content=(
    "You are a text analysis assistant. You can only help with three things: "
    "counting words, counting sentences, and finding the top keywords in text "
    "the user provides. Use word_count, sentence_count and top_keywords for those, "
    "call every tool a question needs, and never estimate numbers. "
    "If the request is about anything else, such as jokes or general questions, "
    "do not call any tools. Reply politely that you can only help with text "
    "analysis, and list what you can do."
))
    reply = model_with_tools.invoke([system] + state["messages"])
    return {"messages": [reply], "llm_calls": state.get("llm_calls", 0) + 1}


def tool_node(state: AgentState):
    """Run every tool the model asked for."""
    results = []
    for call in state["messages"][-1].tool_calls:
        observation = tools_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=call["id"]))
    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tool_node", END]:
    if state["messages"][-1].tool_calls:
        return "tool_node"
    return END


builder = StateGraph(AgentState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()


if __name__ == "__main__":
    text = (
        "SynapseBug is a Flask web app that summarizes text. It uses spaCy for "
        "sentence splitting and RAKE for keyword extraction. Users register and log in. "
        "Admins can delete users."
    )
    question = f"How many words and sentences are in this text, and what are its top 3 keywords?\n\n{text}"
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(result["messages"][-1].text)