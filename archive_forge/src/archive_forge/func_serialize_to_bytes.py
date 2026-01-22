from __future__ import annotations
import logging
import operator
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import (
def serialize_to_bytes(self) -> bytes:
    """Serialize FAISS index, docstore, and index_to_docstore_id to bytes."""
    return pickle.dumps((self.index, self.docstore, self.index_to_docstore_id))