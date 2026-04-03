from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import json

# --- Загружаем конфиг ---
with open("config.json") as f:
    config = json.load(f)

# Request schema
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str


# Response schema
class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str


# Service
class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )


# API
class EmbeddingAPI:
    def __init__(self, model_name: str):
        self.service = EmbeddingService(model_name)
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self):
        self.app.post("/v1/embeddings")(self.create_embeddings)

    def create_embeddings(self, req: EmbeddingRequest):
        # нормализация input
        if isinstance(req.input, str):
            inputs = [req.input]
        else:
            inputs = req.input

        vectors = self.service.embed(inputs)

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": v.tolist(),
                    "index": i
                }
                for i, v in enumerate(vectors)
            ],
            "model": req.model
        }


# Запуск
api = EmbeddingAPI(config['model_name_or_path'])
app = api.app

# Для запуска в консоли 
# /uvicorn Embedder_api:app --reload --host 0.0.0.0 --port 8000

# Использование
# from openai import OpenAI

# client = OpenAI(
#     base_url="http://localhost:8000/v1",
#     api_key="sk-no-key-needed"
# )

# response = client.embeddings.create(
#     model="bge-m3",
#     input="hello world"
# )

# print(response.data[0].embedding)