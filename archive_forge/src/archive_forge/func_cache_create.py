from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def cache_create(self, config: str='') -> requests.Response:
    """Create the cache for the vector db
        Args:
            config(str): configuration of the cache.
        Returns:
            An http Response containing the result of the operation
        """
    if config == '':
        config = '\n            {\n  "distributed-cache": {\n    "owners": "2",\n    "mode": "SYNC",\n    "statistics": true,\n    "encoding": {\n      "media-type": "application/x-protostream"\n    },\n    "indexing": {\n      "enabled": true,\n      "storage": "filesystem",\n      "startup-mode": "AUTO",\n      "indexing-mode": "AUTO",\n      "indexed-entities": [\n        "' + self._entity_name + '"\n      ]\n    }\n  }\n}\n'
    return self.ispn.cache_post(self._cache_name, config)