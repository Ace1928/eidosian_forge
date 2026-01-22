import logging
from typing import Optional
from ray._private import ray_constants
import ray._private.gcs_aio_client
from ray.core.generated.common_pb2 import ErrorType, JobConfig
from ray.core.generated.gcs_pb2 import (
def cleanup_redis_storage(host: str, port: int, password: str, use_ssl: bool, storage_namespace: str):
    """This function is used to cleanup the storage. Before we having
    a good design for storage backend, it can be used to delete the old
    data. It support redis cluster and non cluster mode.

    Args:
       host: The host address of the Redis.
       port: The port of the Redis.
       password: The password of the Redis.
       use_ssl: Whether to encrypt the connection.
       storage_namespace: The namespace of the storage to be deleted.
    """
    from ray._raylet import del_key_from_storage
    if not isinstance(host, str):
        raise ValueError('Host must be a string')
    if not isinstance(password, str):
        raise ValueError('Password must be a string')
    if port < 0:
        raise ValueError(f'Invalid port: {port}')
    if not isinstance(use_ssl, bool):
        raise TypeError('use_ssl must be a boolean')
    if not isinstance(storage_namespace, str):
        raise ValueError('storage namespace must be a string')
    return del_key_from_storage(host, port, password, use_ssl, storage_namespace)