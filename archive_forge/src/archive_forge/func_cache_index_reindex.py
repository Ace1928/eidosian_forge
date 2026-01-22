from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def cache_index_reindex(self) -> requests.Response:
    """Rebuild the for the vector db
        Returns:
            An http Response containing the result of the operation
        """
    return self.ispn.index_reindex(self._cache_name)