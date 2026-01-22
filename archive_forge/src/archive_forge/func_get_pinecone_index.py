from __future__ import annotations
import logging
import os
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Union
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils.iter import batch_iterate
from langchain_core.vectorstores import VectorStore
from packaging import version
from langchain_community.vectorstores.utils import (
@classmethod
def get_pinecone_index(cls, index_name: Optional[str], pool_threads: int=4) -> Index:
    """Return a Pinecone Index instance.

        Args:
            index_name: Name of the index to use.
            pool_threads: Number of threads to use for index upsert.
        Returns:
            Pinecone Index instance."""
    pinecone = _import_pinecone()
    if _is_pinecone_v3():
        pinecone_instance = pinecone.Pinecone(api_key=os.environ.get('PINECONE_API_KEY'), pool_threads=pool_threads)
        indexes = pinecone_instance.list_indexes()
        index_names = [i.name for i in indexes.index_list['indexes']]
    else:
        index_names = pinecone.list_indexes()
    if index_name in index_names:
        index = pinecone_instance.Index(index_name) if _is_pinecone_v3() else pinecone.Index(index_name, pool_threads=pool_threads)
    elif len(index_names) == 0:
        raise ValueError("No active indexes found in your Pinecone project, are you sure you're using the right Pinecone API key and Environment? Please double check your Pinecone dashboard.")
    else:
        raise ValueError(f"Index '{index_name}' not found in your Pinecone project. Did you mean one of the following indexes: {', '.join(index_names)}")
    return index