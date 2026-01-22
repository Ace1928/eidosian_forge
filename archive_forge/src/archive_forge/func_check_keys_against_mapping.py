from abc import ABC, abstractmethod
from typing import AsyncIterable, Tuple
from fastapi import HTTPException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.schemas import chat, completions, embeddings
@classmethod
def check_keys_against_mapping(cls, mapping, payload):
    for k1, k2 in mapping.items():
        if k2 in payload:
            raise HTTPException(status_code=400, detail=f'Invalid parameter {k2}. Use {k1} instead.')