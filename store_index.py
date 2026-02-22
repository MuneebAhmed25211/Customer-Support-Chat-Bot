from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import LoadingData, Embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Get chunks and embeddings from helper
chunks = LoadingData()
embeddings = Embeddings()

# Strip metadata
for chunk in chunks:
    chunk.metadata = {"source": chunk.metadata.get("source", "unknown")}

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

if "myindex" not in pc.list_indexes().names():
    pc.create_index(
        name="myindex",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vectorstore = PineconeVectorStore.from_documents(
    chunks,
    embeddings,
    index_name="myindex"
)