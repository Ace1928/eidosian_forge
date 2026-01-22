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
def similarity_search_with_score(self, query: str, k: int=4, filter: Optional[RedisFilterExpression]=None, return_metadata: bool=True, **kwargs: Any) -> List[Tuple[Document, float]]:
    """Run similarity search with **vector distance**.

        The "scores" returned from this function are the raw vector
        distances from the query vector. For similarity scores, use
        ``similarity_search_with_relevance_scores``.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filter (RedisFilterExpression, optional): Optional metadata filter.
                Defaults to None.
            return_metadata (bool, optional): Whether to return metadata.
                Defaults to True.

        Returns:
            List[Tuple[Document, float]]: A list of documents that are
                most similar to the query with the distance for each document.
        """
    try:
        import redis
    except ImportError as e:
        raise ImportError('Could not import redis python package. Please install it with `pip install redis`.') from e
    if 'score_threshold' in kwargs:
        logger.warning('score_threshold is deprecated. Use distance_threshold instead.' + 'score_threshold should only be used in ' + 'similarity_search_with_relevance_scores.' + 'score_threshold will be removed in a future release.')
    query_embedding = self._embeddings.embed_query(query)
    redis_query, params_dict = self._prepare_query(query_embedding, k=k, filter=filter, with_metadata=return_metadata, with_distance=True, **kwargs)
    try:
        results = self.client.ft(self.index_name).search(redis_query, params_dict)
    except redis.exceptions.ResponseError as e:
        if str(e).split(' ')[0] == 'Syntax':
            raise ValueError('Query failed with syntax error. ' + 'This is likely due to malformation of ' + 'filter, vector, or query argument') from e
        raise e
    docs_with_scores: List[Tuple[Document, float]] = []
    for result in results.docs:
        metadata = {}
        if return_metadata:
            metadata = {'id': result.id}
            metadata.update(self._collect_metadata(result))
        doc = Document(page_content=result.content, metadata=metadata)
        distance = self._calculate_fp_distance(result.distance)
        docs_with_scores.append((doc, distance))
    return docs_with_scores