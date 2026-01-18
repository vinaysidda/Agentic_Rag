from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from tools import wiki, arxiv, rag_tool

llm = ChatOpenAI(temperature=0)

memory_store = {}

def get_agent(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    return initialize_agent(
        tools=[wiki, arxiv, rag_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory_store[session_id],
        verbose=True
    )
