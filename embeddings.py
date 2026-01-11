# embeddings.py
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EmbeddingWrapper:
    def __init__(self):
        self.model_name = "text-embedding-3-small"
        self.embedding_size = 1536

        self._embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            model=self.model_name
        )

    def embed_query(self, text: str) -> list[float]:
        vector = self._embeddings.embed_query(text)

        if not vector or len(vector) != self.embedding_size:
            raise ValueError(
                f"Embedding size mismatch: {len(vector) if vector else None}"
            )

        return vector


embeddings = EmbeddingWrapper()
 

# #embeddings.py
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Qdrant
# from qdrant_client import QdrantClient
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# # Initialize Qdrant client and embeddings
# qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
# VECTOR_SIZE = 1536
# DISTANCE_METRIC = "Cosine"
