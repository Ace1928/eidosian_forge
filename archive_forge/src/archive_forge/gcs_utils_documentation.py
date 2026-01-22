import logging
from typing import Optional
from ray._private import ray_constants
import ray._private.gcs_aio_client
from ray.core.generated.common_pb2 import ErrorType, JobConfig
from ray.core.generated.gcs_pb2 import (
This function is used to cleanup the storage. Before we having
    a good design for storage backend, it can be used to delete the old
    data. It support redis cluster and non cluster mode.

    Args:
       host: The host address of the Redis.
       port: The port of the Redis.
       password: The password of the Redis.
       use_ssl: Whether to encrypt the connection.
       storage_namespace: The namespace of the storage to be deleted.
    