#ap.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from extractor import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_csv,
    extract_text_from_json,
    extract_text_from_xlsx
)

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from embeddings import embeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

import os
import uuid
import json
import logging
import openai
from datetime import datetime

#APP INIT 
app = Flask(__name__)
CORS(app)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY]):
    raise ValueError("Missing environment variables")

#QDRANT
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

COLLECTION_NAME = "Document"

# Create collection if not exists
if not qdrant_client.collection_exists(COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embeddings.embedding_size,
            distance=models.Distance.COSINE
        )
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

#FILES
UPLOAD_FOLDER = "uploaded_files"
TEMP_FOLDER = "temp_uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ROUTES
@app.route("/upload", methods=["POST"])
def upload_file():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    uploaded = []
    for file in request.files.getlist("files"):
        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        uploaded.append(filename)

    return jsonify({"uploaded": uploaded})


@app.route("/documents", methods=["GET"])
def list_documents():
    docs = []
    for f in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, f)
        docs.append({
            "filename": f,
            "size_kb": round(os.path.getsize(path) / 1024, 2)
        })
    return jsonify(docs)

@app.route("/list_files", methods=["GET"])
def list_files():
    try:
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            limit=1000
        )

        files = {}
        for point in points:
            payload = point.payload or {}
            filename = payload.get("filename")

            if filename and filename not in files:
                files[filename] = {
                    "filename": filename,
                    "description": payload.get("description", ""),
                    "upload_date": payload.get("upload_date", "")
                }

        return jsonify({
            "files": list(files.values())
        }), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to list files",
            "details": str(e)
        }), 500


@app.route("/save_vector", methods=["POST"])
def save_vector():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")

    # Collect descriptions
    descriptions = []
    i = 0
    while True:
        desc = request.form.get(f"descriptions_{i}")
        if desc is None:
            break
        descriptions.append(desc)
        i += 1

    if len(files) != len(descriptions):
        return jsonify({"error": "Files and descriptions mismatch"}), 400

    extractors = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".txt": extract_text_from_txt,
        ".csv": extract_text_from_csv,
        ".json": extract_text_from_json,
        ".xlsx": extract_text_from_xlsx,
    }

    success = []
    failed = []

    for index, file in enumerate(files):
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in extractors:
            failed.append({"filename": filename, "error": "Unsupported file type"})
            continue

        temp_path = os.path.join(TEMP_FOLDER, filename)
        file.save(temp_path)

        try:
            # 🔑 ALWAYS pass file object, not path
            with open(temp_path, "rb") as f:
                text = extractors[ext](f)

            if not text or not text.strip():
                raise ValueError("Extracted text is empty")

            combined_text = f"Description: {descriptions[index]}\n\n{text}"

            chunks = text_splitter.split_text(combined_text)

            if not chunks:
                raise ValueError("Text splitter returned no chunks")

            points = []
            for chunk in chunks:
                vector = embeddings.embed_query(chunk)

                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "filename": filename,
                            "description": descriptions[index],
                            "upload_date": datetime.utcnow().isoformat()
                        }
                    )
                )

            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )

            success.append({
                "filename": filename,
                "chunks_inserted": len(points)
            })

        except Exception as e:
            failed.append({
                "filename": filename,
                "error": str(e)
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return jsonify({
        "success": success,
        "failed": failed
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query_text", "").strip()
    target_files = data.get("target_files", [])

    if not query:
        return jsonify({"error": "Query required"}), 400

    if isinstance(target_files, str):
        target_files = [target_files]

    query_vector = embeddings.embed_query(query)

    q_filter = None
    if target_files:
        q_filter = Filter(
            must=[FieldCondition(
                key="filename",
                match=MatchAny(any=target_files)
            )]
        )

    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=q_filter,
        limit=20
    )

    if not results:
        return jsonify({"error": "No results found"}), 404

    rerank_inputs = [(query, r.payload["text"]) for r in results]
    scores = reranker.predict(rerank_inputs)

    for i, r in enumerate(results):
        r.payload["rerank_score"] = float(scores[i])

    top = sorted(results, key=lambda x: x.payload["rerank_score"], reverse=True)[:10]

    context = "\n\n".join(r.payload["text"] for r in top)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ],
        temperature=0.3,
        max_tokens=800
    )

    return jsonify({
        "answer": response.choices[0].message.content.strip(),
        "sources": [
            {
                "filename": r.payload["filename"],
                "score": r.payload["rerank_score"]
            } for r in top
        ]
    })


@app.route("/delete_file", methods=["POST"])
def delete_file():
    filename = request.json.get("filename")
    if not filename:
        return jsonify({"error": "Filename required"}), 400

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=filename)
                )]
            )
        )
    )

    return jsonify({"message": f"{filename} deleted"})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

