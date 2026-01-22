from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def update_document(self, document_id: str, document: Document) -> None:
    """Update an existing document in the vectorstore."""
    self.vlite.update(document_id, text=document.page_content, metadata=document.metadata)