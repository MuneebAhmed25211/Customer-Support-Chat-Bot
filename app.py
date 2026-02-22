from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pinecone import Pinecone
from src.helper import Embeddings
from src.prompt import memory_prompt
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



# Store chat history per session
chat_histories = {}

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

class AnswerResponse(BaseModel):
    answer: str
    session_id: str

@app.get("/")
def home():
    return {"status": "Customer Support Chatbot is running"}

@app.post("/chat")
def chat(request: QuestionRequest):
    session_id = request.session_id

    # Get or create history for this session
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]

    # Retrieve context from vector DB
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(request.question)
    context = "\n\n".join([d.page_content for d in docs])

    # Build chain with memory
    chain = memory_prompt | llm | StrOutputParser()

    # Invoke with history
    answer = chain.invoke({
        "context": context,
        "chat_history": history,
        "question": request.question
    })

    # Update history
    history.append(HumanMessage(content=request.question))
    history.append(AIMessage(content=answer))

    # Keep last 10 messages only to avoid token overflow
    if len(history) > 10:
        chat_histories[session_id] = history[-10:]

    return AnswerResponse(answer=answer, session_id=session_id)

@app.delete("/chat/{session_id}")
def clear_history(session_id: str):
    if session_id in chat_histories:
        del chat_histories[session_id]
    return {"status": "History cleared", "session_id": session_id}