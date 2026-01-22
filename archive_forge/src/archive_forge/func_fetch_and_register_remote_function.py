import dis
import hashlib
import importlib
import inspect
import json
import logging
import os
import threading
import time
import traceback
from collections import defaultdict, namedtuple
from typing import Optional, Callable
import ray
import ray._private.profiling as profiling
from ray import cloudpickle as pickle
from ray._private import ray_constants
from ray._private.inspect_util import (
from ray._private.ray_constants import KV_NAMESPACE_FUNCTION_TABLE
from ray._private.utils import (
from ray._private.serialization import pickle_dumps
from ray._raylet import (
def fetch_and_register_remote_function(self, key):
    """Import a remote function."""
    remote_function_info = self.fetch_registered_method(key)
    if not remote_function_info:
        return False
    job_id_str, function_id_str, function_name, serialized_function, module, max_calls = remote_function_info
    function_id = ray.FunctionID(function_id_str)
    job_id = ray.JobID(job_id_str)
    max_calls = int(max_calls)
    with self.lock:
        self._num_task_executions[function_id] = 0
        try:
            function = pickle.loads(serialized_function)
        except Exception:
            traceback_str = format_error_message(traceback.format_exc())

            def f(*args, **kwargs):
                raise RuntimeError('The remote function failed to import on the worker. This may be because needed library dependencies are not installed in the worker environment:\n\n{}'.format(traceback_str))
            self._function_execution_info[function_id] = FunctionExecutionInfo(function=f, function_name=function_name, max_calls=max_calls)
            logger.debug(f"Failed to unpickle the remote function '{function_name}' with function ID {function_id.hex()}. Job ID:{job_id}.Traceback:\n{traceback_str}. ")
        else:
            function.__module__ = module
            self._function_execution_info[function_id] = FunctionExecutionInfo(function=function, function_name=function_name, max_calls=max_calls)
    return True