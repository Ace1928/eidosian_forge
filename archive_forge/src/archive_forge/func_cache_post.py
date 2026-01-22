from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def cache_post(self, name: str, config: str) -> requests.Response:
    """Create a cache
        Args:
            name(str): name of the cache.
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
    api_url = self._default_node + self._cache_url + '/' + name
    response = requests.post(api_url, config, headers={'Content-Type': 'application/json'}, timeout=REST_TIMEOUT)
    return response