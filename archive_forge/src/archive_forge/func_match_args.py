from __future__ import annotations
import uuid
import warnings
from itertools import repeat
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def match_args(self, query: List[float], filter: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ret: Dict[str, Any] = dict(query_embedding=query)
    if filter:
        ret['filter'] = filter
    return ret