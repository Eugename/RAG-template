from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import json

class EmbeddingRequest(BaseModel):
    input: list[str]

class EmbeddingAPI:
    def __init__(self, config: dict):
        self.config = config
        self.app = FastAPI()
        self._load_model()
        self._register_routes()

    def _load_model(self):
        self.model = SentenceTransformer(
            self.config["model_name_or_path"]
        )

    def _register_routes(self):
        self.app.post("/embeddings")(self.embed)

    def embed(self, req: EmbeddingRequest):
        vectors = self.model.encode(
            req.input,
            normalize_embeddings=True
        )
        return {
            "data": [{"embedding": v.tolist()} for v in vectors]
        }


# --- запуск ---
with open("config.json") as f:
    config = json.load(f)

api = EmbeddingAPI(config)
app = api.app