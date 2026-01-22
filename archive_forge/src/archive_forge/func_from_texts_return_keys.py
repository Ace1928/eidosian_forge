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
def from_texts_return_keys(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]]=None, index_name: Optional[str]=None, index_schema: Optional[Union[Dict[str, ListOfDict], str, os.PathLike]]=None, vector_schema: Optional[Dict[str, Union[str, int]]]=None, **kwargs: Any) -> Tuple[Redis, List[str]]:
    """Create a Redis vectorstore from raw documents.

        This is a user-friendly interface that:
            1. Embeds documents.
            2. Creates a new Redis index if it doesn't already exist
            3. Adds the documents to the newly created Redis index.
            4. Returns the keys of the newly created documents once stored.

        This method will generate schema based on the metadata passed in
        if the `index_schema` is not defined. If the `index_schema` is defined,
        it will compare against the generated schema and warn if there are
        differences. If you are purposefully defining the schema for the
        metadata, then you can ignore that warning.

        To examine the schema options, initialize an instance of this class
        and print out the schema using the `Redis.schema`` property. This
        will include the content and content_vector classes which are
        always present in the langchain schema.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import Redis
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings()
                redis, keys = Redis.from_texts_return_keys(
                    texts,
                    embeddings,
                    redis_url="redis://localhost:6379"
                )

        Args:
            texts (List[str]): List of texts to add to the vectorstore.
            embedding (Embeddings): Embeddings to use for the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadata
                dicts to add to the vectorstore. Defaults to None.
            index_name (Optional[str], optional): Optional name of the index to
                create or add to. Defaults to None.
            index_schema (Optional[Union[Dict[str, ListOfDict], str, os.PathLike]],
                optional):
                Optional fields to index within the metadata. Overrides generated
                schema. Defaults to None.
            vector_schema (Optional[Dict[str, Union[str, int]]], optional): Optional
                vector schema to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the Redis client.

        Returns:
            Tuple[Redis, List[str]]: Tuple of the Redis instance and the keys of
                the newly created documents.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
        """
    try:
        import redis
        from langchain_community.vectorstores.redis.schema import read_schema
    except ImportError as e:
        raise ImportError('Could not import redis python package. Please install it with `pip install redis`.') from e
    redis_url = get_from_dict_or_env(kwargs, 'redis_url', 'REDIS_URL')
    if 'redis_url' in kwargs:
        kwargs.pop('redis_url')
    if 'generate' in kwargs:
        kwargs.pop('generate')
    keys = None
    if 'keys' in kwargs:
        keys = kwargs.pop('keys')
    if not index_name:
        index_name = uuid.uuid4().hex
    if metadatas:
        if isinstance(metadatas, list) and len(metadatas) != len(texts):
            raise ValueError('Number of metadatas must match number of texts')
        if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
            raise ValueError('Metadatas must be a list of dicts')
        generated_schema = _generate_field_schema(metadatas[0])
        if index_schema:
            user_schema = read_schema(index_schema)
            if user_schema != generated_schema:
                logger.warning('`index_schema` does not match generated metadata schema.\n' + 'If you meant to manually override the schema, please ' + 'ignore this message.\n' + f'index_schema: {user_schema}\n' + f'generated_schema: {generated_schema}\n')
        else:
            index_schema = generated_schema
    instance = cls(redis_url, index_name, embedding, index_schema=index_schema, vector_schema=vector_schema, **kwargs)
    keys = instance.add_texts(texts, metadatas, keys=keys)
    return (instance, keys)