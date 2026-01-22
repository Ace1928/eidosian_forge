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
def future(self) -> Future:
    fut = Future()

    def set_future(data: Any) -> None:
        """Schedules a callback to set the exception or result
            in the Future."""
        if isinstance(data, Exception):
            fut.set_exception(data)
        else:
            fut.set_result(data)
    self._on_completed(set_future)
    fut.object_ref = self
    return fut