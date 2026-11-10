# Text Analysis Agent

An AI agent built with LangGraph and LangChain that analyzes text. The LLM decides which tool to call (word count, sentence count, top keywords), and a FastAPI endpoint exposes it.

## How it works
1. The user sends text and a question to `POST /analyze`.
2. The LLM decides whether a tool is needed, and the graph loops between the model and the tools until it can answer.
3. The response includes the answer, the list of tools used, and the number of LLM calls, so results can be audited.
4. Off-topic requests are declined without calling any tool.

## Tech stack
Python, FastAPI, LangGraph, LangChain, Google Gemini, Pydantic

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and add your Google AI Studio API key.

## Run
```
uvicorn agent_api:app --reload
```
Open http://127.0.0.1:8000/docs and try `POST /analyze`.

## Example
POST /analyze
{"text": "Flask is a web framework. It is simple.", "question": "How many words and sentences are there?"}

## Limitations and next steps
- Keyword ranking is by frequency only (next: TF-IDF).
- No automated tests yet (next: pytest checks against known counts).
- Add a document-search tool that queries a RAG service.