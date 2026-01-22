from __future__ import annotations
import pickle
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def get_documents_array_uri(uri: str) -> str:
    """Get the URI of the documents array."""
    return f'{uri}/{DOCUMENTS_ARRAY_NAME}'