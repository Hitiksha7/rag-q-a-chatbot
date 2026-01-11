from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# ---------------- LOAD ENV ----------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Document")

# ---------------- CLIENT ----------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

# ---------------- CREATE PAYLOAD INDEX ----------------
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="filename",
    field_schema=PayloadSchemaType.KEYWORD
)

print("Payload index created for 'filename'")
