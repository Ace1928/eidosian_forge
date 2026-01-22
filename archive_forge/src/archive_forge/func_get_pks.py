from __future__ import annotations
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def get_pks(self, expr: str, **kwargs: Any) -> List[int] | None:
    """Get primary keys with expression

        Args:
            expr: Expression - E.g: "id in [1, 2]", or "title LIKE 'Abc%'"

        Returns:
            List[int]: List of IDs (Primary Keys)
        """
    from pymilvus import MilvusException
    if self.col is None:
        logger.debug('No existing collection to get pk.')
        return None
    try:
        query_result = self.col.query(expr=expr, output_fields=[self._primary_field])
    except MilvusException as exc:
        logger.error('Failed to get ids: %s error: %s', self.collection_name, exc)
        raise exc
    pks = [item.get(self._primary_field) for item in query_result]
    return pks