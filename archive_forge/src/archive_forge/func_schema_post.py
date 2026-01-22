from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def schema_post(self, name: str, proto: str) -> requests.Response:
    """Deploy a schema
        Args:
            name(str): name of the schema. Will be used as a key
            proto(str): protobuf schema
        Returns:
            An http Response containing the result of the operation
        """
    api_url = self._default_node + self._schema_url + '/' + name
    response = requests.post(api_url, proto, timeout=REST_TIMEOUT)
    return response