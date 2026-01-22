import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def create_metadata(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Create metadata from fields.

    Args:
        fields: The fields of the document. The fields must be a dict.

    Returns:
        metadata: The metadata of the document. The metadata must be a dict.
    """
    metadata: Dict[str, Any] = {}
    for key, value in fields.items():
        if key == 'id' or key == 'document' or key == 'embedding':
            continue
        metadata[key] = value
    return metadata