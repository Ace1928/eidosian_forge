from __future__ import annotations
import logging
from copy import deepcopy
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def delete_texts(self, ids: List[str]) -> None:
    """Delete a list of docs from the Rockset collection"""
    try:
        from rockset.models import DeleteDocumentsRequestData
    except ImportError:
        raise ImportError('Could not import rockset client python package. Please install it with `pip install rockset`.')
    self._client.Documents.delete_documents(collection=self._collection_name, data=[DeleteDocumentsRequestData(id=i) for i in ids], workspace=self._workspace)