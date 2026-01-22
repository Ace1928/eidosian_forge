from __future__ import annotations
import os
import pickle
import uuid
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def dependable_annoy_import() -> Any:
    """Import annoy if available, otherwise raise error."""
    try:
        import annoy
    except ImportError:
        raise ImportError('Could not import annoy python package. Please install it with `pip install --user annoy` ')
    return annoy