from __future__ import annotations
import enum
import logging
import uuid
from datetime import timedelta
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
@classmethod
def service_url_from_db_params(cls, host: str, port: int, database: str, user: str, password: str) -> str:
    """Return connection string from database parameters."""
    return f'postgresql://{user}:{password}@{host}:{port}/{database}'