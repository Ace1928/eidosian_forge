from __future__ import annotations
import uuid
import warnings
from typing import (
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def create_knn_index(self, mapping: Dict) -> None:
    """
        Create a new k-NN index in Elasticsearch.

        Args:
            mapping (Dict): The mapping to use for the new index.

        Returns:
            None
        """
    self.client.indices.create(index=self.index_name, mappings=mapping)