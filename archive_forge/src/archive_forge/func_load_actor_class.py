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
def load_actor_class(self, job_id, actor_creation_function_descriptor):
    """Load the actor class.
        Args:
            job_id: job ID of the actor.
            actor_creation_function_descriptor: Function descriptor of
                the actor constructor.
        Returns:
            The actor class.
        """
    function_id = actor_creation_function_descriptor.function_id
    actor_class = self._loaded_actor_classes.get(function_id, None)
    if actor_class is None:
        if self._worker.load_code_from_local:
            actor_class = self._load_actor_class_from_local(actor_creation_function_descriptor)
            if actor_class is None:
                actor_class = self._load_actor_class_from_gcs(job_id, actor_creation_function_descriptor)
        else:
            actor_class = self._load_actor_class_from_gcs(job_id, actor_creation_function_descriptor)
        self._loaded_actor_classes[function_id] = actor_class
        module_name = actor_creation_function_descriptor.module_name
        actor_class_name = actor_creation_function_descriptor.class_name
        actor_methods = inspect.getmembers(actor_class, predicate=is_function_or_method)
        for actor_method_name, actor_method in actor_methods:
            if actor_method_name == '__init__':
                method_descriptor = actor_creation_function_descriptor
            else:
                method_descriptor = PythonFunctionDescriptor(module_name, actor_method_name, actor_class_name)
            method_id = method_descriptor.function_id
            executor = self._make_actor_method_executor(actor_method_name, actor_method, actor_imported=True)
            self._function_execution_info[method_id] = FunctionExecutionInfo(function=executor, function_name=actor_method_name, max_calls=0)
            self._num_task_executions[method_id] = 0
        self._num_task_executions[function_id] = 0
    return actor_class