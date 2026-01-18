from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from os import getenv
load_dotenv()


# 1. Load documents
# loader = PyPDFLoader("Data/Democratizing_AI_access.pdf")
# docs = loader.load()

import os
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "..", "Data", "Democratizing_AI_access.pdf")

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# 3. Embeddings + Vector Store
embeddings = OpenAIEmbeddings(api_key=getenv("OPENAI_API_KEY"))
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. LLM
llm = ChatOpenAI(temperature=0)

def answer_from_docs(question: str) -> str:
    """Traditional RAG function"""
    retrieved_docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join(d.page_content for d in retrieved_docs)

    prompt = f"""
    Answer using ONLY the context below.
    Context:
    {context}

    Question:
    {question}
    """

    return llm.invoke(prompt).content
