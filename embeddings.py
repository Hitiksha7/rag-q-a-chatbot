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
 

