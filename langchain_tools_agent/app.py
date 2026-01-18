from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tools import wiki, arxiv, rag_tool
from agent import get_agent

app = FastAPI(title="3-Tool Agentic RAG")

# app.mount("/static", StaticFiles(directory="static"), name="static")

class RagRequest(BaseModel):
    question: str

class AgentRequest(BaseModel):
    session_id: str
    message: str

@app.get("/wiki")
def wiki_api(query: str):
    return {
        "source": "wikipedia",
        "answer": wiki.run(query)
    }

@app.get("/arxiv")
def arxiv_api(query: str):
    return {
        "source": "arxiv",
        "answer": arxiv.run(query)
    }

@app.post("/rag")
def rag_api(req: RagRequest):
    return {
        "source": "internal_docs",
        "answer": rag_tool.run(req.question)
    }

@app.post("/agent")
def agent_api(req: AgentRequest):
    agent = get_agent(req.session_id)
    return {
        "answer": agent.run(req.message)
    }
