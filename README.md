Semantic PDF Search with RAG, Embeddings, and Chroma DB
Demo project for Retrieval-Augmented Generation (RAG) using LLM embeddings, chunking, and local vector database search.

Overview
This repository is a minimal but powerful example of an end-to-end semantic search workflow.
You can drop any PDF(s) into the data/ folder and instantly perform question-answering over your documents using modern embeddings and a local vector database (Chroma).

Language: Python

Libraries: LangChain, ChromaDB, Sentence Transformers

Concepts: RAG (Retrieval Augmented Generation), Text Chunking, Embeddings, Semantic Search, Vector DB

Features
Loads and chunks 1+ PDFs into overlapping segments for granular search.

Embeds all chunks with all-MiniLM-L6-v2 (Sentence Transformers) for fast, high-quality semantic representations.

Stores and queries using ChromaDB.

CLI interface lets you interactively ask any question—instantly retrieves the top semantic matches.

Architecture
text
flowchart LR
    A[PDF Files] --> B[Chunking & Overlap]
    B --> C[Embedding (MiniLM)]
    C --> D[Vector DB (Chroma)]
    E[User Query] --> F[Embedding & Top-k Search]
    F --> D
    D --> G[Answer: Retrieved Chunks]
How it Works
Drop PDF(s) – Place documents in /data.

Run Script – Python loads, splits, embeds, and builds vector DB.

Query – Type any question in the CLI. The app finds the top semantic chunks and displays them instantly.

Sample Usage
bash
python src/main.py
# Type your question (or 'exit' to quit):

Q: what is overfitting?
File Structure
text
semantic-search-pdf/
├── data/           # Place PDFs here
├── src/            # Python source code
|   └── main.py
├── requirements.txt
├── README.md
└── .gitignore
Key Interview Talking Points
Chunking: Granular segments (500 chars, 100 overlap) improve retrieval and LLM context.

Embeddings: Uses a state-of-the-art transformer, so semantically similar queries always match.

Vector DB: Chroma enables efficient similarity search and integrates easily for projects/production.

Professional Workflow: venv, requirements.txt, .gitignore, modular code—critical software engineering habits.

Next Steps
Add a Streamlit or Flask web UI for demo.

Plug in LLM (OpenAI, local models) for auto-generating answers from retrieved context.

Explore advanced chunking strategies and metadata-aware retrieval.