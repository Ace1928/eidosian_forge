from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def create_index_if_not_exist(self, dim: int, distance_type: str, index_type: str, data_type: str, **kwargs: Any) -> bool:
    index = self.client.tvs_get_index(self.index_name)
    if index is not None:
        logger.info('Index already exists')
        return False
    self.client.tvs_create_index(self.index_name, dim, distance_type, index_type, data_type, **kwargs)
    return True