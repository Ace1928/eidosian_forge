from fastapi import FastAPI, HTTPException
import uvicorn
import httpx
import os
from pydantic import BaseModel
from typing import List, Union, Dict, Any

try:
    from eidosian_core.ports import get_service_port, get_service_url
except Exception:  # pragma: no cover
    def get_service_port(*_args, default: int = 11435, **_kwargs) -> int:
        return default

    def get_service_url(*_args, default_port: int = 11434, default_path: str = "", **_kwargs) -> str:
        return f"http://127.0.0.1:{default_port}{default_path}"

app = FastAPI()

DEFAULT_PROXY_PORT = get_service_port(
    "file_forge_embedding_proxy",
    default=11435,
    env_keys=("EIDOS_FILE_FORGE_EMBED_PROXY_PORT",),
)
DEFAULT_OLLAMA_URL = os.environ.get(
    "OLLAMA_URL",
    get_service_url("ollama_http", default_port=11434, default_path=""),
)
OLLAMA_URL = DEFAULT_OLLAMA_URL

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str

class EmbeddingResponse(BaseModel):
    object: str
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, Any]

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    async with httpx.AsyncClient() as client:
        if isinstance(request.input, str):
            request.input = [request.input]

        ollama_requests = [{"model": request.model, "prompt": text} for text in request.input]

        embeddings: List[Dict[str, Any]] = []

        for i, ollama_request in enumerate(ollama_requests):
            response = await client.post(f"{OLLAMA_URL}/api/embeddings", json=ollama_request)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Ollama API error")
            
            result = response.json()
            embeddings.append({
                "object": "embedding",
                "embedding": result["embedding"],
                "index": i
            })
            
        return EmbeddingResponse(
            object="list",
            data=embeddings,
            model=request.model,
            usage={"prompt_tokens": 0, "total_tokens": 0}  # Simple placeholder for usage stats
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the embedding proxy server")
    parser.add_argument("--port", type=int, default=DEFAULT_PROXY_PORT, help="Port to run the server on")
    parser.add_argument("--host", type=str, default=DEFAULT_OLLAMA_URL, help="URL of the Ollama server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    OLLAMA_URL = args.host
    uvicorn.run("embedding_proxy:app", host="0.0.0.0", port=args.port, reload=args.reload)
