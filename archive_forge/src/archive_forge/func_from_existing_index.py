from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def from_existing_index(cls, embedding: Embeddings, index_name: str, schema: Union[Dict[str, ListOfDict], str, os.PathLike, Dict[str, ListOfDict]], key_prefix: Optional[str]=None, **kwargs: Any) -> Redis:
    """Connect to an existing Redis index.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Redis
                from langchain_community.embeddings import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()

                # must pass in schema and key_prefix from another index
                existing_rds = Redis.from_existing_index(
                    embeddings,
                    index_name="my-index",
                    schema=rds.schema, # schema dumped from another index
                    key_prefix=rds.key_prefix, # key prefix from another index
                    redis_url="redis://username:password@localhost:6379",
                )

        Args:
            embedding (Embeddings): Embedding model class (i.e. OpenAIEmbeddings)
                for embedding queries.
            index_name (str): Name of the index to connect to.
            schema (Union[Dict[str, str], str, os.PathLike, Dict[str, ListOfDict]]):
                Schema of the index and the vector schema. Can be a dict, or path to
                yaml file.
            key_prefix (Optional[str]): Prefix to use for all keys in Redis associated
                with this index.
            **kwargs (Any): Additional keyword arguments to pass to the Redis client.

        Returns:
            Redis: Redis VectorStore instance.

        Raises:
            ValueError: If the index does not exist.
            ImportError: If the redis python package is not installed.
        """
    redis_url = get_from_dict_or_env(kwargs, 'redis_url', 'REDIS_URL')
    if 'redis_url' in kwargs:
        kwargs.pop('redis_url')
    instance = cls(redis_url, index_name, embedding, index_schema=schema, key_prefix=key_prefix, **kwargs)
    if not check_index_exists(instance.client, index_name):
        raise ValueError(f'Redis failed to connect: Index {index_name} does not exist.')
    return instance