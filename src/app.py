"""Flask Web Application for Semantic PDF Search

This module provides a web interface for the RAG pipeline, allowing users to:
- Search uploaded PDFs semantically
- Upload new PDF documents
- View search results with similarity scores

Features:
- RESTful API endpoints for search and upload
- CORS-enabled for cross-origin requests
- Static file serving (CSS, JavaScript)
- Integration with ChromaDB for vector search

Author: Abdur Rab Dad Khan
Date: 2025
Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ChromaDB for vector search
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ---- CONFIGURATION ----
DB_DIR = Path(__file__).parent.parent / "chroma_db"
DATA_DIR = Path(__file__).parent.parent / "data"
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# ---- FLASK APP INITIALIZATION ----
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS for development
CORS(app)

logger.info("Initializing Flask application...")
logger.info(f"Template folder: {app.template_folder}")
logger.info(f"Static folder: {app.static_folder}")

# ---- VECTOR DATABASE INITIALIZATION ----
logger.info(f"Initializing embedder model: {EMBEDDING_MODEL}")
embedder = SentenceTransformerEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}  # Use "cuda" if GPU available
)

logger.info(f"Loading ChromaDB from: {DB_DIR}")
db = Chroma(
    collection_name="pdf_chunks",
    embedding_function=embedder,
    persist_directory=str(DB_DIR)
)
logger.info("ChromaDB initialized successfully")


# ---- API ENDPOINTS ----

@app.route("/", methods=["GET"])
def home():
    """
    Serve the main HTML interface.
    
    Returns:
        HTML: The main search interface template
    """
    logger.info("Serving home page")
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search() -> Dict[str, Any]:
    """
    Semantic search endpoint for PDF chunks.
    
    Request JSON:
        - query (str): User's search query
        - k (int, optional): Number of results (default: 3)
    
    Returns:
        JSON: List of search results with:
            - text: Chunk content
            - similarity: Similarity score (0-100)
            - metadata: Document metadata (page, source)
    
    Example:
        POST /api/search
        {"query": "What is machine learning?", "k": 5}
        
        Response:
        [
            {
                "text": "Machine learning is...",
                "similarity": 92.5,
                "metadata": {"page": 1, "source": "document.pdf"}
            },
            ...
        ]
    """
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        k = data.get("k", 3)
        
        # Validate input
        if not user_query:
            logger.warning("Empty query received")
            return jsonify({"error": "Query cannot be empty"}), 400
        
        if k < 1 or k > 10:
            k = 3  # Default to 3
        
        logger.info(f"Processing search query: '{user_query}' (k={k})")
        
        # Perform similarity search with scores
        results = db.similarity_search_with_scores(user_query, k=k)
        
        if not results:
            logger.info(f"No results found for query: {user_query}")
            return jsonify([]), 200
        
        # Format results for response
        formatted_results = []
        for doc, score in results:
            similarity_percent = (1 - score) * 100  # Convert distance to similarity %
            result = {
                "text": doc.page_content[:400],  # Limit text length
                "similarity": round(similarity_percent, 2),
                "metadata": doc.metadata or {}
            }
            formatted_results.append(result)
        
        logger.info(f"Returned {len(formatted_results)} results")
        return jsonify(formatted_results), 200
        
    except Exception as e:
        logger.exception(f"Error in search endpoint: {str(e)}")
        return jsonify({"error": f"Search failed: {str(e)}"}), 500


@app.route("/api/upload", methods=["POST"])
def upload_pdf() -> tuple[str, int]:
    """
    Upload a new PDF file to the data directory.
    
    NOTE: Re-indexing the database is required after upload.
    Run 'python src/main.py' to rebuild the vector database.
    
    Returns:
        tuple: (message, status_code)
            - 200: File uploaded successfully
            - 400: No file provided or invalid format
            - 413: File too large
    
    Example:
        POST /api/upload
        files: {"pdf": <file>}
    """
    try:
        # Check if file is in request
        if "pdf" not in request.files:
            logger.warning("Upload request with no file")
            return jsonify({"error": "No file provided"}), 400
        
        pdf_file = request.files["pdf"]
        
        # Check file is selected
        if pdf_file.filename == "":
            logger.warning("Upload request with empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file extension
        if not allowed_file(pdf_file.filename):
            logger.warning(f"Upload rejected: invalid file type - {pdf_file.filename}")
            return jsonify({"error": "Only PDF files are allowed"}), 400
        
        # Save file
        filename = secure_filename(pdf_file.filename)
        filepath = DATA_DIR / filename
        pdf_file.save(str(filepath))
        
        logger.info(f"PDF file uploaded successfully: {filename}")
        
        return jsonify({
            "message": f"PDF uploaded: {filename}",
            "note": "Run 'python src/main.py' to rebuild the search index"
        }), 200
        
    except Exception as e:
        logger.exception(f"Error in upload endpoint: {str(e)}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring.
    
    Returns:
        JSON: Status information
    """
    try:
        # Try a simple query to verify DB connection
        db.similarity_search("test", k=1)
        status = "healthy"
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        status = "unhealthy"
    
    return jsonify({
        "status": status,
        "database": "ChromaDB",
        "embedding_model": EMBEDDING_MODEL
    }), 200


# ---- UTILITY FUNCTIONS ----

def allowed_file(filename: str) -> bool:
    """
    Check if uploaded file has allowed extension.
    
    Args:
        filename (str): Name of the file
    
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---- ERROR HANDLERS ----

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


# ---- MAIN EXECUTION ----

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("Starting Semantic PDF Search Web Server")
    logger.info("="*70)
    logger.info(f"Database directory: {DB_DIR}")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("Server running on http://localhost:5000")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*70)
    
    # Run Flask development server
    # Use debug=False in production
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=True
    )
