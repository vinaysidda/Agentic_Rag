from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from os import getenv
load_dotenv()


loader = PyPDFLoader("Data\Democratizing.pdf")
document = loader.load()

# chunk document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 30
)


splitted_text = text_splitter.split_documents(documents=document)

# embeeding + vector store 

embedding = OpenAIEmbeddings(api_key=getenv("OPENAI_API_KEY"))
vector_Store =FAISS.from_documents(embedding,text_splitter)

retriver = vector_Store.as_retriever(search_type ="similarity",kwargs={"k":5})

query = "what is langchain"
retrived_docs = retriver.get_relevant_documents(query)

llm = ChatOpenAI(temperature=0)


from langchain.chains import retrieval_qa

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(temperature=0)

qa_chain = retrieval_qa.from_chain_type(
    llm=llm,
    retriever=retriver,
    chain_type="stuff"
)

response = qa_chain.run(query)



retriver = vector_Store.as_retriever(search_type = "similarity",kwargs={"k":3})

llm = ChatOpenAI(model="gpt-4o",temperature =0) # use the good opena ai model 


retrived_answers = retrieval_qa.from_Chain_type(
    llm =llm ,
    retriever = retriver,
    chain_type = "stuff",
    return_source_documents=True,

)
 
response = retrived_answers.invoke(query)
