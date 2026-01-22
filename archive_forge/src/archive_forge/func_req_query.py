from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def req_query(self, query: str, cache_name: str, local: bool=False) -> requests.Response:
    """Request a query
        Args:
            query(str): query requested
            cache_name(str): name of the target cache
            local(boolean): whether the query is local to clustered
        Returns:
            An http Response containing the result set or errors
        """
    if self._use_post_for_query:
        return self._query_post(query, cache_name, local)
    return self._query_get(query, cache_name, local)