
# 🤖 RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents in multiple formats and ask questions about their content using OpenAI's GPT-4 and vector embeddings.


## ✨ Features

- **Multi-format Support**: PDF, DOCX, TXT, CSV, JSON, XLSX
- **Semantic Search**: Vector embeddings using OpenAI's text-embedding-3-small (1536 dimensions)
- **Re-ranking**: CrossEncoder for improved accuracy
- **Conversational AI**: GPT-4o-mini for intelligent responses
- **Document Management**: Upload, view, and delete documents
- **User-friendly Interface**: Clean Streamlit UI
- **Fast Retrieval**: Qdrant vector database with cosine similarity search

## 🏗️ Architecture
<img width="272" height="340" alt="Image" src="https://github.com/user-attachments/assets/1c9ef1df-30b9-4ff1-b163-c86e32b4f102" />

#### Document processing flow
<img width="683" height="60" alt="Image" src="https://github.com/user-attachments/assets/91c9a188-9c6b-4679-b1fd-950472f62015" />

#### Question-answering flow
<img width="688" height="55" alt="Image" src="https://github.com/user-attachments/assets/d7856de6-4363-4117-8c65-243404cf6ad7" />

### How It Works

1. **Document Upload**: User uploads documents with descriptions
2. **Text Extraction**: Content extracted based on file type (PyPDF2, python-docx, pandas)
3. **Chunking**: Text split into 1000-character chunks with 200-character overlap
4. **Embedding**: Each chunk converted to 1536-dimensional vector
5. **Storage**: Vectors stored in Qdrant with metadata (filename, description, upload date)
6. **Query Processing**:
   - User question embedded to same vector space
   - Top 20 similar chunks retrieved via cosine similarity
   - CrossEncoder re-ranks to top 10 most relevant
   - Context passed to GPT-4 for answer generation
 
## 🛠️ Technology Stack

### Core Technologies
- **Python**: Primary programming language
- **Flask**: REST API backend framework
- **Streamlit**: Web UI framework
- **Qdrant**: Vector database

### AI/ML Libraries
- **OpenAI API**: text-embedding-3-small (embeddings), GPT-4o-mini (generation)
- **sentence-transformers**: CrossEncoder for re-ranking
- **LangChain**: Text splitting utilities

### Document Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: Microsoft Word document handling
- **pandas**: CSV file processing
- **openpyxl**: Excel file support
- **json**: JSON document parsing

### Infrastructure
- **python-dotenv**: Environment variable management
- **Flask-CORS**: Cross-origin resource sharing
- **requests**: HTTP client for frontend-backend communication

## 📝 API Endpoints

### Backend API (Flask - Port 5000)

| Endpoint | Method | Description | Parameters |
|----------|--------|-------------|------------|
| `/save_vector` | POST | Upload and embed documents | `files` (multipart), `descriptions_N` (form data) |
| `/chat` | POST | Query documents | `query_text` (string), `target_files` (array) |
| `/list_files` | GET | Retrieve all documents | None |
| `/delete_file` | POST | Remove document and vectors | `filename` (string) |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant instance URL | `https://xyz.qdrant.io` |
| `QDRANT_API_KEY` | Qdrant authentication key | `your_key_here` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
