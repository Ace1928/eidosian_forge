from __future__ import annotations
import logging
from typing import (
import numpy as np
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def from_connection_string(cls, connection_string: str, namespace: str, embedding: Embeddings, **kwargs: Any) -> MongoDBAtlasVectorSearch:
    """Construct a `MongoDB Atlas Vector Search` vector store
        from a MongoDB connection URI.

        Args:
            connection_string: A valid MongoDB connection URI.
            namespace: A valid MongoDB namespace (database and collection).
            embedding: The text embedding model to use for the vector store.

        Returns:
            A new MongoDBAtlasVectorSearch instance.

        """
    try:
        from importlib.metadata import version
        from pymongo import MongoClient
        from pymongo.driver_info import DriverInfo
    except ImportError:
        raise ImportError('Could not import pymongo, please install it with `pip install pymongo`.')
    client: MongoClient = MongoClient(connection_string, driver=DriverInfo(name='Langchain', version=version('langchain')))
    db_name, collection_name = namespace.split('.')
    collection = client[db_name][collection_name]
    return cls(collection, embedding, **kwargs)