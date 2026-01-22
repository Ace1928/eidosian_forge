from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def similarity_search_by_vector(self, embedding: List[float], k: int=4, filter: Optional[RedisFilterExpression]=None, return_metadata: bool=True, distance_threshold: Optional[float]=None, **kwargs: Any) -> List[Document]:
    """Run similarity search between a query vector and the indexed vectors.

        Args:
            embedding (List[float]): The query vector for which to find similar
                documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.
            distance_threshold (Optional[float], optional): Maximum vector distance
                between selected documents and the query vector. Defaults to None.

        Returns:
            List[Document]: A list of documents that are most similar to the query
                text.
        """
    try:
        import redis
    except ImportError as e:
        raise ImportError('Could not import redis python package. Please install it with `pip install redis`.') from e
    if 'score_threshold' in kwargs:
        logger.warning('score_threshold is deprecated. Use distance_threshold instead.' + 'score_threshold should only be used in ' + 'similarity_search_with_relevance_scores.' + 'score_threshold will be removed in a future release.')
    redis_query, params_dict = self._prepare_query(embedding, k=k, filter=filter, distance_threshold=distance_threshold, with_metadata=return_metadata, with_distance=False)
    try:
        results = self.client.ft(self.index_name).search(redis_query, params_dict)
    except redis.exceptions.ResponseError as e:
        if str(e).split(' ')[0] == 'Syntax':
            raise ValueError('Query failed with syntax error. ' + 'This is likely due to malformation of ' + 'filter, vector, or query argument') from e
        raise e
    docs = []
    for result in results.docs:
        metadata = {}
        if return_metadata:
            metadata = {'id': result.id}
            metadata.update(self._collect_metadata(result))
        content_key = self._schema.content_key
        docs.append(Document(page_content=getattr(result, content_key), metadata=metadata))
    return docs