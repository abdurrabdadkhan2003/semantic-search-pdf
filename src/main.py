"""Semantic PDF Search - Core RAG Pipeline

This module implements the complete RAG (Retrieval-Augmented Generation) pipeline:
1. Load PDFs from the data/ directory
2. Split documents into overlapping chunks for context preservation
3. Generate semantic embeddings using Sentence Transformers
4. Store embeddings in ChromaDB for efficient similarity search
5. Provide interactive CLI for semantic queries

Author: Abdur Rab Dad Khan
Date: 2025
Version: 1.0.0
"""

from pathlib import Path
import logging
from typing import List

# Configure logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- PDF LOADING & CHUNKING ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---- VECTOR DB ----
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ---- CONFIGURATION ----
# Path configuration - relative to project root
DATA_DIR = Path(__file__).parent.parent / "data"
DB_DIR = Path(__file__).parent.parent / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# ---- EMBEDDING MODEL CONFIGURATION ----
# Using all-MiniLM-L6-v2: 22MB, 95% accuracy, 10x faster than larger models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- CHUNKING CONFIGURATION ----
CHUNK_SIZE = 500  # Characters per chunk (optimal for LLM context windows)
CHUNK_OVERLAP = 100  # Character overlap (prevents losing context at chunk boundaries)
SEPARATORS = ["\n\n", "\n", " ", ""]  # Hierarchical splitting: paragraphs -> sentences -> chars


def load_documents() -> List[Document]:
    """
    Load all PDF files from the 'data' directory and extract their text.
    
    Returns:
        List[Document]: A list of LangChain Document objects (one per page).
                       Each document contains page_content and metadata (filename, page).
    
    Raises:
        FileNotFoundError: If data directory doesn't exist or contains no PDFs.
    
    Example:
        >>> docs = load_documents()
        >>> print(f"Loaded {len(docs)} pages")
        Loaded 50 pages
    """
    logger.info(f"Searching for PDFs in: {DATA_DIR}")
    pdf_paths = list(DATA_DIR.glob("*.pdf"))
    
    if not pdf_paths:
        logger.warning(f"No PDF files found in {DATA_DIR}")
        return []
    
    logger.info(f"Found {len(pdf_paths)} PDF file(s)")
    docs = []
    
    for pdf_path in pdf_paths:
        logger.info(f"Loading: {pdf_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            logger.info(f"  ✓ Extracted {len(pages)} pages from {pdf_path.name}")
            docs.extend(pages)
        except Exception as e:
            logger.error(f"  ✗ Error loading {pdf_path.name}: {str(e)}")
    
    logger.info(f"Total pages loaded: {len(docs)}")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller overlapping chunks for RAG.
    
    Chunking Strategy:
    - Chunk Size: 500 characters (optimal balance between context and precision)
    - Overlap: 100 characters (prevents important info from being split across chunks)
    - Separators: Hierarchical (paragraphs → sentences → characters)
    
    Args:
        docs (List[Document]): Documents loaded from PDFs.
    
    Returns:
        List[Document]: Chunked documents with preserved metadata.
    
    Example:
        >>> chunks = chunk_documents(docs)
        >>> print(f"Created {len(chunks)} chunks")
        Created 12000 chunks
    """
    logger.info(f"Chunking {len(docs)} documents...")
    logger.info(f"Parameters: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    
    chunks = splitter.split_documents(docs)
    logger.info(f"✓ Created {len(chunks)} chunks from {len(docs)} documents")
    
    return chunks


def build_vector_db(chunks: List[Document]) -> Chroma:
    """
    Embed chunks and build a persistent ChromaDB vector database.
    
    Embedding Model: all-MiniLM-L6-v2
    - Size: 22MB (lightweight)
    - Speed: ~8000 docs/sec
    - Quality: 95% of larger models like BGE-Large
    - Format: Can be exported to ONNX for production
    
    Args:
        chunks (List[Document]): Chunked documents.
    
    Returns:
        Chroma: Persistent vector database instance.
    
    Raises:
        ValueError: If chunks list is empty.
    
    Example:
        >>> db = build_vector_db(chunks)
        >>> print("Vector database built successfully")
        Vector database built successfully
    """
    if not chunks:
        raise ValueError("Cannot build vector DB with empty chunks list")
    
    logger.info(f"Initializing Sentence Transformer embeddings...")
    logger.info(f"Model: {EMBEDDING_MODEL}")
    
    # Initialize embedder - this downloads the model on first run (~22MB)
    embedder = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # Use GPU if available: "cuda"
    )
    
    logger.info("Creating ChromaDB collection...")
    # Initialize Chroma - creates persistent collection
    db = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embedder,
        persist_directory=str(DB_DIR),
    )
    
    logger.info(f"Embedding and storing {len(chunks)} chunks...")
    # Extract texts and metadata
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    
    # Add to database
    db.add_texts(texts=texts, metadatas=metadatas)
    db.persist()  # Save to disk
    
    logger.info(f"✓ Successfully embedded and stored {len(texts)} chunks in ChromaDB!")
    logger.info(f"Database location: {DB_DIR}")
    
    return db


def query_loop(db: Chroma) -> None:
    """
    Interactive CLI for semantic search over PDF chunks.
    
    Features:
    - Real-time semantic search using embeddings
    - Returns top-3 most relevant chunks
    - Shows snippet of each result (first 300 chars)
    - Type 'exit' to quit
    
    Args:
        db (Chroma): Vector database instance.
    
    Example:
        >>> query_loop(db)
        Type your question (or 'exit' to quit):
        Q: What is machine learning?
        [1] Machine learning is a subset of artificial intelligence...
    """
    print("\n" + "="*70)
    print("SEMANTIC PDF SEARCH - Interactive Query Mode")
    print("="*70)
    print("\nTip: Ask questions about your PDF documents.")
    print("The system will find the most relevant sections.\n")
    
    while True:
        user_query = input("\nQ: ").strip()
        
        if not user_query:
            print("Please enter a query.")
            continue
        
        if user_query.lower() in ["exit", "quit", "q"]:
            print("\nThank you for using Semantic PDF Search!")
            break
        
        logger.info(f"Processing query: {user_query}")
        
        # Perform similarity search
        results = db.similarity_search_with_scores(user_query, k=3)
        
        if not results:
            print("\nNo relevant results found.")
            continue
        
        # Display results
        print("\n" + "-"*70)
        for idx, (doc, score) in enumerate(results, 1):
            snippet = doc.page_content[:300].replace('\n', ' ')
            similarity = f"{(1-score)*100:.1f}%"  # Convert distance to similarity %
            print(f"\n[Result {idx}] (Similarity: {similarity})")
            print(f"{snippet}...")
            
            # Show source if available in metadata
            if 'source' in doc.metadata:
                print(f"Source: {doc.metadata['source']}")
            if 'page' in doc.metadata:
                print(f"Page: {doc.metadata['page']}")
        
        print("\n" + "-"*70)


def main() -> None:
    """
    Main execution pipeline.
    
    Steps:
    1. Load PDFs from data/ directory
    2. Split into overlapping chunks
    3. Generate embeddings and build vector DB
    4. Launch interactive query interface
    """
    logger.info("="*70)
    logger.info("Starting Semantic PDF Search RAG Pipeline")
    logger.info("="*70)
    
    try:
        # Step 1: Load PDFs
        logger.info("\n[Step 1/4] Loading PDF documents...")
        docs = load_documents()
        
        if not docs:
            logger.error("No documents loaded. Please add PDFs to the 'data' folder.")
            return
        
        logger.info(f"Loaded {len(docs)} document pages from PDF(s).")
        
        # Step 2: Chunk PDFs
        logger.info("\n[Step 2/4] Chunking documents...")
        chunks = chunk_documents(docs)
        logger.info(f"Chunked into {len(chunks)} RAG-ready segments!")
        
        # Display example chunk
        print("\nExample chunk text:")
        print("-" * 70)
        print(chunks[0].page_content[:500])
        print("-" * 70)
        
        # Step 3: Build Vector DB
        logger.info("\n[Step 3/4] Building vector database...")
        db = build_vector_db(chunks)
        
        # Step 4: Launch Query Interface
        logger.info("\n[Step 4/4] Launching interactive query interface...")
        query_loop(db)
        
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        raise
    
    logger.info("Pipeline completed successfully!")


# ---- ENTRY POINT ----
if __name__ == "__main__":
    main()
