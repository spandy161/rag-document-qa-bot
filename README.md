# 📄 RAG-Based Document Q&A Bot

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based Question-Answering system that allows users to query a collection of documents and receive **grounded, context-aware answers with source citations**.

The system combines:

* **Semantic search (retrieval)** from a vector database
* **Local LLM (generation)** using Ollama
* **Strict grounding rules** to prevent hallucinations

---

## 🎯 Features

* 📂 Load and process PDF documents
* ✂️ Intelligent text chunking with overlap
* 🔢 Embedding generation using lightweight models
* 🧠 Semantic search with vector database (ChromaDB)
* 🤖 Local LLM inference (Ollama – Phi3)
* 📌 Source-aware answers with citations
* 🚫 Hallucination prevention ("I don't know" fallback)
* 💻 CLI-based interactive interface

---

## 🧠 System Architecture

```
User Query
   ↓
Embedding (MiniLM)
   ↓
Vector DB (Chroma)
   ↓
Top-K Retrieval
   ↓
Context Construction
   ↓
LLM (Ollama - Phi3)
   ↓
Final Answer + Source Citation
```

---

## ⚙️ Technical Decisions

### 1. Document Ingestion

* PDFs loaded using LangChain loaders
* Extracted text cleaned and structured
* Metadata stored:

  * Source file name
  * Page number

---

### 2. Text Chunking

* Strategy: **RecursiveCharacterTextSplitter**
* Chunk size: ~500–800 tokens
* Overlap: ~50–100 tokens

**Reason:**
Preserves semantic continuity and avoids context loss at chunk boundaries.

---

### 3. Embeddings

* Model: `all-MiniLM-L6-v2`

**Reason:**

* Lightweight and fast
* Good semantic performance
* Works locally (no API cost)

---

### 4. Vector Database

* Database: **ChromaDB**
* Persistent storage (`/db` folder)

**Reason:**

* Simple setup
* Local persistence
* Efficient similarity search

---

### 5. Retrieval

* Method: **Similarity search with threshold filtering**
* Top-K retrieval: 2–4 chunks

**Enhancements:**

* Relevance filtering
* Context validation
* Query normalization (for better matching)

---

### 6. Answer Generation

* Model: **Ollama (Phi3)**

**Why local LLM?**

* No API cost
* Offline capability
* Faster iteration

---

### 7. Hallucination Control (IMPORTANT)

To ensure reliability:

* Model is **forced to answer only from context**
* If answer not found → returns:

  ```
  I don't know.
  ```
* Prevents use of external knowledge

---

## 💻 How to Run

### 1. Clone Repository

```bash
git clone <github.com/spandy161/rag-document-qa-bot>
cd RAG_QABOT
```

---

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Ingestion

```bash
python ingest.py
```

---

### 5. Start Q&A System

```bash
python query.py
```

---

## 🧪 Example Usage

```
Ask: What is RAG?

Answer:
RAG stands for Retrieval-Augmented Generation, a method that combines retrieval of external knowledge with language models to improve accuracy.

Sources:
- rag.pdf (Page 1)
```

---

```
Ask: What is IPL cricket?

Answer:
I don't know.
```

---

## 📁 Project Structure

```
RAG_QABOT/
│
├── data/                # Input documents (PDFs)
├── db/                  # Vector database (persisted)
├── ingest.py            # Document ingestion + embedding
├── query.py             # Query + retrieval + generation
├── requirements.txt
├── README.md
└── .env.example
```

---

## 📌 Assignment Requirements Coverage

| Requirement                | Status |
| -------------------------- | ------ |
| Document ingestion         | ✅      |
| Text chunking              | ✅      |
| Embeddings                 | ✅      |
| Vector DB persistence      | ✅      |
| Retrieval (Top-K)          | ✅      |
| Grounded answer generation | ✅      |
| Source citation            | ✅      |
| CLI interface              | ✅      |

---

## 🎥 Demo

A screen recording demonstrating:

* Application startup
* Multiple queries
* Answer generation
* Source citation

(https://www.loom.com/share/3db094ef79ca48c4bad5f6a35fc4f47d)

---

## ⚠️ Limitations

* Retrieval depends on chunk quality
* Some queries may return "I don't know" if context is insufficient
* Local LLM may produce shorter or conservative answers

---

## 🚀 Future Improvements

* Better retrieval (reranking)
* Hybrid search (keyword + semantic)
* Web UI (Streamlit)
* Multi-document summarization
* Improved chunking strategies

---

## 👤 Author

* Name: Spandan Mendhe
* Role: AI Intern Candidate

---

## 📌 Conclusion

This project demonstrates a **complete end-to-end RAG pipeline** with strong emphasis on:

* Accuracy
* Grounding
* Reliability

Designed to minimize hallucination while maintaining useful responses.
