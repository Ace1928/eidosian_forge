from __future__ import annotations
import copy
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
from langchain_core.documents import BaseDocumentTransformer, Document
def split_documents(self, documents: Iterable[Document]) -> List[Document]:
    """Split documents."""
    texts, metadatas = ([], [])
    for doc in documents:
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)
    return self.create_documents(texts, metadatas=metadatas)