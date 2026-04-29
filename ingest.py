import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "data"
DB_PATH = "db"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()
            
            # Add filename metadata
            for doc in docs:
                doc.metadata["source"] = file
            
            documents.extend(docs)
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)

def main():
    print("🔄 Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages")

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("🔢 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Storing in vector DB...")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )

    db.persist()
    print("✅ Indexing complete!")

if __name__ == "__main__":
    main()