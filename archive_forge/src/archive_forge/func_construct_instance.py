from __future__ import annotations
import functools
import uuid
import warnings
from itertools import islice
from operator import itemgetter
from typing import (
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def construct_instance(cls: Type[Qdrant], texts: List[str], embedding: Embeddings, location: Optional[str]=None, url: Optional[str]=None, port: Optional[int]=6333, grpc_port: int=6334, prefer_grpc: bool=False, https: Optional[bool]=None, api_key: Optional[str]=None, prefix: Optional[str]=None, timeout: Optional[float]=None, host: Optional[str]=None, path: Optional[str]=None, collection_name: Optional[str]=None, distance_func: str='Cosine', content_payload_key: str=CONTENT_KEY, metadata_payload_key: str=METADATA_KEY, vector_name: Optional[str]=VECTOR_NAME, shard_number: Optional[int]=None, replication_factor: Optional[int]=None, write_consistency_factor: Optional[int]=None, on_disk_payload: Optional[bool]=None, hnsw_config: Optional[common_types.HnswConfigDiff]=None, optimizers_config: Optional[common_types.OptimizersConfigDiff]=None, wal_config: Optional[common_types.WalConfigDiff]=None, quantization_config: Optional[common_types.QuantizationConfig]=None, init_from: Optional[common_types.InitFrom]=None, on_disk: Optional[bool]=None, force_recreate: bool=False, **kwargs: Any) -> Qdrant:
    try:
        import qdrant_client
    except ImportError:
        raise ValueError('Could not import qdrant-client python package. Please install it with `pip install qdrant-client`.')
    from grpc import RpcError
    from qdrant_client.http import models as rest
    from qdrant_client.http.exceptions import UnexpectedResponse
    partial_embeddings = embedding.embed_documents(texts[:1])
    vector_size = len(partial_embeddings[0])
    collection_name = collection_name or uuid.uuid4().hex
    distance_func = distance_func.upper()
    client, async_client = cls._generate_clients(location=location, url=url, port=port, grpc_port=grpc_port, prefer_grpc=prefer_grpc, https=https, api_key=api_key, prefix=prefix, timeout=timeout, host=host, path=path, **kwargs)
    try:
        if force_recreate:
            raise ValueError
        collection_info = client.get_collection(collection_name=collection_name)
        current_vector_config = collection_info.config.params.vectors
        if isinstance(current_vector_config, dict) and vector_name is not None:
            if vector_name not in current_vector_config:
                raise QdrantException(f'Existing Qdrant collection {collection_name} does not contain vector named {vector_name}. Did you mean one of the existing vectors: {', '.join(current_vector_config.keys())}? If you want to recreate the collection, set `force_recreate` parameter to `True`.')
            current_vector_config = current_vector_config.get(vector_name)
        elif isinstance(current_vector_config, dict) and vector_name is None:
            raise QdrantException(f'Existing Qdrant collection {collection_name} uses named vectors. If you want to reuse it, please set `vector_name` to any of the existing named vectors: {', '.join(current_vector_config.keys())}.If you want to recreate the collection, set `force_recreate` parameter to `True`.')
        elif not isinstance(current_vector_config, dict) and vector_name is not None:
            raise QdrantException(f"Existing Qdrant collection {collection_name} doesn't use named vectors. If you want to reuse it, please set `vector_name` to `None`. If you want to recreate the collection, set `force_recreate` parameter to `True`.")
        if current_vector_config.size != vector_size:
            raise QdrantException(f'Existing Qdrant collection is configured for vectors with {current_vector_config.size} dimensions. Selected embeddings are {vector_size}-dimensional. If you want to recreate the collection, set `force_recreate` parameter to `True`.')
        current_distance_func = current_vector_config.distance.name.upper()
        if current_distance_func != distance_func:
            raise QdrantException(f'Existing Qdrant collection is configured for {current_distance_func} similarity, but requested {distance_func}. Please set `distance_func` parameter to `{current_distance_func}` if you want to reuse it. If you want to recreate the collection, set `force_recreate` parameter to `True`.')
    except (UnexpectedResponse, RpcError, ValueError):
        vectors_config = rest.VectorParams(size=vector_size, distance=rest.Distance[distance_func], on_disk=on_disk)
        if vector_name is not None:
            vectors_config = {vector_name: vectors_config}
        client.recreate_collection(collection_name=collection_name, vectors_config=vectors_config, shard_number=shard_number, replication_factor=replication_factor, write_consistency_factor=write_consistency_factor, on_disk_payload=on_disk_payload, hnsw_config=hnsw_config, optimizers_config=optimizers_config, wal_config=wal_config, quantization_config=quantization_config, init_from=init_from, timeout=timeout)
    qdrant = cls(client=client, collection_name=collection_name, embeddings=embedding, content_payload_key=content_payload_key, metadata_payload_key=metadata_payload_key, distance_strategy=distance_func, vector_name=vector_name, async_client=async_client)
    return qdrant