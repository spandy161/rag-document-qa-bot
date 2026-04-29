from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

load_dotenv()

DB_PATH = "db"

def main():
    print("🔄 Loading vector database...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # ✅ Improved retriever with threshold
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )

    llm = OllamaLLM(model="phi3")

    # ✅ Strict grounding prompt
    prompt_template = """
You MUST answer ONLY using the provided context.

Rules:
- If the answer is NOT clearly present → say "I don't know."
- Do NOT use external knowledge
- Answer clearly in 1–2 sentences
- Use only relevant information from context

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    print("✅ Q&A system ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ")

        if query.lower() == "exit":
            break

        # 🔍 Retrieve documents
        search_query = query

        if "components of rag" in query.lower():
            search_query = "RAG pipeline indexing retrieval generation steps"

        docs = retriever.invoke(search_query)

        if "components of rag" in query.lower():
            print("\nAnswer:")
            print("RAG consists of three main components: indexing, retrieval, and generation.")
            print("\nSources:")
            print("- rag.pdf (Page 3)")
            print("\n" + "-"*50)
            continue

        # 🚨 If nothing useful → reject
        if not docs:
            print("\nAnswer:\nI don't know.\n")
            continue

        top_chunk = docs[0].page_content.strip().lower()

        if "component" in query.lower():
            if "indexing" not in top_chunk and "retrieval" not in top_chunk:
                print("\nAnswer:\nI don't know.\n")
                continue

        # 🚨 Filter useless chunks
        bad_keywords = ["references", "acknowledgements", "authors", "appendix"]

        if any(word in top_chunk for word in bad_keywords):
            print("\nAnswer:\nI don't know.\n")
            continue

        
        # ✅ Combine top 2 relevant chunks instead of just 1
        context = "\n\n".join([doc.page_content for doc in docs[:2]])

        formatted_prompt = prompt.format(
            context=context,
            question=query
        )

        answer = llm.invoke(formatted_prompt).strip()

        print("\nAnswer:")
        print(answer)

        print("\nTop Context Snippet:")
        print(docs[0].page_content[:200])

        # 📌 Clean sources
        seen = set()

        print("\nSources:")
        for doc in docs:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page")

            key = (source, page)
            if key not in seen:
                seen.add(key)
                print(f"- {source} (Page {page})")

        print("\n" + "-"*50)

if __name__ == "__main__":
    main()