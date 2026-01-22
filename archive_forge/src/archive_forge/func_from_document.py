from __future__ import annotations
import hashlib
import json
import uuid
from itertools import islice
from typing import (
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.vectorstores import VectorStore
from langchain.indexes.base import NAMESPACE_UUID, RecordManager
@classmethod
def from_document(cls, document: Document, *, uid: Optional[str]=None) -> _HashedDocument:
    """Create a HashedDocument from a Document."""
    return cls(uid=uid, page_content=document.page_content, metadata=document.metadata)