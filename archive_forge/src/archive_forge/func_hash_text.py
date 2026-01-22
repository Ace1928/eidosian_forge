import hashlib
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.retrievers import BaseRetriever
def hash_text(text: str) -> str:
    """Hash a text using SHA256.

    Args:
        text: Text to hash.

    Returns:
        Hashed text.
    """
    return str(hashlib.sha256(text.encode('utf-8')).hexdigest())