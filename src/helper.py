from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def LoadingData():
    loader = DirectoryLoader("data/", glob="*.pdf",
    loader_cls=PyPDFLoader)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)

    chunks = splitter.split_documents(docs)
    return chunks

def Embeddings():
    embeddings = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"

)   
    return embeddings