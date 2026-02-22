from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pinecone import Pinecone
from src.helper import Embeddings
from src.prompt import prompt

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = Embeddings()
pc = Pinecone(api_key=PINECONE_API_KEY)

vectorstore = PineconeVectorStore(
    index=pc.Index("myindex"),
    embedding=embeddings
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

rag_chain = (
    {
        "context": vectorstore.as_retriever(search_kwargs={"k": 2})
                   | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | prompt | llm | StrOutputParser()
)

# ── FastAPI ────────────────────────────────
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/")
def home():
    return {"status": "Customer Support Chatbot is running"}

@app.post("/chat")
def chat(request: QuestionRequest):
    answer = rag_chain.invoke(request.question)
    return AnswerResponse(answer=answer)