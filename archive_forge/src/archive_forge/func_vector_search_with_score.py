from __future__ import annotations
import base64
import json
import logging
import uuid
from typing import (
import numpy as np
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
def vector_search_with_score(self, query: str, k: int=4, filters: Optional[str]=None) -> List[Tuple[Document, float]]:
    """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query and score for each
        """
    from azure.search.documents.models import VectorizedQuery
    results = self.client.search(search_text='', vector_queries=[VectorizedQuery(vector=np.array(self.embed_query(query), dtype=np.float32).tolist(), k_nearest_neighbors=k, fields=FIELDS_CONTENT_VECTOR)], filter=filters, top=k)
    docs = [(Document(page_content=result.pop(FIELDS_CONTENT), metadata=json.loads(result[FIELDS_METADATA]) if FIELDS_METADATA in result else {k: v for k, v in result.items() if k != FIELDS_CONTENT_VECTOR}), float(result['@search.score'])) for result in results]
    return docs