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
def deserialize_objects(self, data_metadata_pairs, object_refs):
    assert len(data_metadata_pairs) == len(object_refs)
    if not hasattr(self._thread_local, 'object_ref_stack'):
        self._thread_local.object_ref_stack = []
    results = []
    for object_ref, (data, metadata) in zip(object_refs, data_metadata_pairs):
        try:
            self._thread_local.object_ref_stack.append(object_ref)
            obj = self._deserialize_object(data, metadata, object_ref)
        except Exception as e:
            logger.exception(e)
            obj = RaySystemError(e, traceback.format_exc())
        finally:
            if self._thread_local.object_ref_stack:
                self._thread_local.object_ref_stack.pop()
        results.append(obj)
    return results