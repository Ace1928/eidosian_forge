from __future__ import annotations
import json
import uuid
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def bulk_similarity_search(self, queries: Iterable[Union[str, Dict[str, float]]], k: int=4, **kwargs: Any) -> List[List[Document]]:
    """Search the marqo index for the most similar documents in bulk with multiple
        queries.

        Args:
            queries (Iterable[Union[str, Dict[str, float]]]): An iterable of queries to
            execute in bulk, queries in the list can be strings or dictionaries of
            weighted queries.
            k (int, optional): The number of documents to return for each query.
            Defaults to 4.

        Returns:
            List[List[Document]]: A list of results for each query.
        """
    bulk_results = self.marqo_bulk_similarity_search(queries=queries, k=k)
    bulk_documents: List[List[Document]] = []
    for results in bulk_results['result']:
        documents = self._construct_documents_from_results_without_score(results)
        bulk_documents.append(documents)
    return bulk_documents