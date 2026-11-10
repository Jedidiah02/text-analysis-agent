from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.messages import HumanMessage

from text_agent import agent

app = FastAPI(title="Text Analysis Agent")


class AnalyzeRequest(BaseModel):
    text: str
    question: str = "How many words and sentences are in this text, and what are its top 5 keywords?"


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    prompt = f"{req.question}\n\nText:\n{req.text}"
    try:
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    except Exception:
        raise HTTPException(status_code=502, detail="The language model service failed. Please try again.")

    tools_used = [
        call["name"]
        for m in result["messages"]
        if getattr(m, "tool_calls", None)
        for call in m.tool_calls
    ]
    return {
        "answer": result["messages"][-1].text,
        "tools_used": tools_used,
        "llm_calls": result["llm_calls"],
    }