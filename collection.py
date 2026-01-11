from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

COLLECTION_NAME = "Document"

# Delete old collection (384-dim)
try:
    client.delete_collection(COLLECTION_NAME)
    print("Old collection deleted")
except Exception:
    print("Collection did not exist")

# Create new collection (1536-dim)
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE
    )
)

print("Collection recreated with 1536 dimensions")
