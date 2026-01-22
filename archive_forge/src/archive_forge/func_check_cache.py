import inspect
import logging
import os
import pickle
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import grpc
import ray._raylet as raylet
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.signature import extract_signature, get_signature
from ray._private.utils import check_oversized_function
from ray.util.client import ray
from ray.util.client.options import validate_options
def check_cache(self, req_id: int) -> Optional[Any]:
    """
        Check the cache for a given thread, and see if the entry in the cache
        matches the current request_id. Returns None if the request_id has
        not been seen yet, otherwise returns the cached result.
        """
    with self.cv:
        if _id_is_newer(self.last_received, req_id) or self.last_received == req_id:
            raise RuntimeError(f'Attempting to accesss a cache entry that has already cleaned up. The client has already acknowledged receiving this response. ({req_id}, {self.last_received})')
        if req_id in self.cache:
            cached_resp = self.cache[req_id]
            while cached_resp is None:
                self.cv.wait()
                if req_id not in self.cache:
                    raise RuntimeError('Cache entry was removed. This likely means that the result of this call is no longer needed.')
                cached_resp = self.cache[req_id]
            return cached_resp
        self.cache[req_id] = None
    return None