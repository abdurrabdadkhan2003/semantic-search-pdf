from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from pathlib import Path
from werkzeug.utils import secure_filename

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DB_DIR = Path(__file__).parent.parent / "chroma_db"
DATA_DIR = Path(__file__).parent.parent / "data"

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

embedder = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    collection_name="pdf_chunks",
    embedding_function=embedder,
    persist_directory=str(DB_DIR)
)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    user_query = request.json.get("query", "")
    results = db.similarity_search(user_query, k=3)
    data = [
        {"text": res.page_content[:400]} for res in results
    ]
    return jsonify(data)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    pdf_file = request.files.get('pdf')
    if pdf_file:
        path = DATA_DIR / secure_filename(pdf_file.filename)
        pdf_file.save(str(path))
        return "PDF uploaded. Re-run index script to update search.", 200
    return "No file.", 400

if __name__ == "__main__":
    app.run(debug=True)
