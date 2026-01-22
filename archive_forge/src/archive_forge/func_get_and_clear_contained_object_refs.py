import io
import logging
import threading
import traceback
from typing import Any
import google.protobuf.message
import ray._private.utils
import ray.cloudpickle as pickle
from ray._private import ray_constants
from ray._raylet import (
from ray.core.generated.common_pb2 import ErrorType, RayErrorInfo
from ray.exceptions import (
from ray.util import serialization_addons
from ray.util import inspect_serializability
def get_and_clear_contained_object_refs(self):
    if not hasattr(self._thread_local, 'object_refs'):
        self._thread_local.object_refs = set()
        return set()
    object_refs = self._thread_local.object_refs
    self._thread_local.object_refs = set()
    return object_refs