from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from rag import answer_from_docs

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=3))
arxiv = ArxivQueryRun(
    api_wrapper=ArxivAPIWrapper(max_results=3)
    )    

# RAG Tool
@tool
def rag_tool(question: str) -> str:
    """
    Use this tool for answering questions from internal documents only.
    """
    return answer_from_docs(question)cd 