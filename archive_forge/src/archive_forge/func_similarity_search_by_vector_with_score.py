from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
def similarity_search_by_vector_with_score(self, query_embedding: List[float], k: int=4, **kwargs: Any) -> List[Tuple[Document, float]]:
    """
        Performs similarity search from a embeddings vector.

        Args:
            query_embedding: Embeddings vector to search for.
            k: Number of results to return.
            custom_query: Use this custom query instead default query (kwargs)
            kwargs: other vector store specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
    if 'custom_query' in kwargs:
        query = kwargs['custom_query']
    else:
        query = self._create_query(query_embedding, k, **kwargs)
    try:
        response = self._vespa_app.query(body=query)
    except Exception as e:
        raise RuntimeError(f'Could not retrieve data from Vespa: {e.args[0][0]['summary']}. Error: {e.args[0][0]['message']}')
    if not str(response.status_code).startswith('2'):
        raise RuntimeError(f'Could not retrieve data from Vespa. Error code: {response.status_code}. Message: {response.json['message']}')
    root = response.json['root']
    if 'errors' in root:
        import json
        raise RuntimeError(json.dumps(root['errors']))
    if response is None or response.hits is None:
        return []
    docs = []
    for child in response.hits:
        page_content = child['fields'][self._page_content_field]
        score = child['relevance']
        metadata = {'id': child['id']}
        if self._metadata_fields is not None:
            for field in self._metadata_fields:
                metadata[field] = child['fields'].get(field)
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append((doc, score))
    return docs