from pathlib import Path

# ---- PDF LOADING & CHUNKING ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- VECTOR DB ----
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DB_DIR = Path(__file__).parent.parent / "chroma_db"

def load_documents():
    """
    Loads all PDF files from the 'data' directory and extracts their text.
    Returns a list of LangChain Document objects (one per page).
    """
    pdf_paths = list(DATA_DIR.glob("*.pdf"))
    docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    return docs

def chunk_documents(docs):
    """
    Splits pages into smaller overlapping chunks for RAG.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def build_vector_db(chunks):
    """
    Embeds each chunk and builds a persistent Chroma vector DB.
    """
    # CORRECT embedding wrapper — REQUIRED for Chroma
    embedder = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embedder,
        persist_directory=str(DB_DIR)
    )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    db.add_texts(texts=texts, metadatas=metadatas)
    db.persist()

    print(f"Embedded and stored {len(texts)} chunks in Chroma DB!")
    return db

def query_loop(db):
    """
    Interactive CLI for semantic search over chunks.
    """
    print("\nType your question (or 'exit' to quit):")
    while True:
        query = input("\nQ: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        results = db.similarity_search(query, k=3)
        for idx, res in enumerate(results, 1):
            snippet = res.page_content[:300].replace('\n', ' ')
            print(f"\n[{idx}] {snippet} ...\n---")

# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    # 1. Load PDFs
    docs = load_documents()
    print(f"Loaded {len(docs)} document pages from PDF(s).")

    # 2. Chunk PDFs
    chunks = chunk_documents(docs)
    print(f"Chunked into {len(chunks)} RAG-ready segments!\n")
    print("Example chunk text:\n")
    print(chunks[0].page_content[:500])

    # 3. Build Vector DB
    db = build_vector_db(chunks)

    # 4. Semantic search CLI
    query_loop(db)