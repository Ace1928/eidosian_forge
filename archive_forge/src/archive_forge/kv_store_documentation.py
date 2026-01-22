import logging
from typing import Optional
import ray
from ray._private import ray_constants
from ray._raylet import GcsClient
from ray.serve._private.constants import RAY_SERVE_KV_TIMEOUT_S, SERVE_LOGGER_NAME
from ray.serve._private.storage.kv_store_base import KVStoreBase
Delete the value associated with the given key from the store.

        Args:
            key (str)
        