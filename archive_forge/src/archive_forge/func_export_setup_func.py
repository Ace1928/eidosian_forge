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
def export_setup_func(self, setup_func: Callable, timeout: Optional[int]=None) -> bytes:
    """Export the setup hook function and return the key."""
    pickled_function = pickle_dumps(setup_func, f'Cannot serialize the worker_process_setup_hook {setup_func.__name__}')
    function_to_run_id = hashlib.shake_128(pickled_function).digest(ray_constants.ID_SIZE)
    key = make_function_table_key(WORKER_PROCESS_SETUP_HOOK_KEY_NAME_GCS.encode(), self._worker.current_job_id.binary(), function_to_run_id)
    check_oversized_function(pickled_function, setup_func.__name__, 'function', self._worker)
    try:
        self._worker.gcs_client.internal_kv_put(key, pickle.dumps({'job_id': self._worker.current_job_id.binary(), 'function_id': function_to_run_id, 'function': pickled_function}), True, ray_constants.KV_NAMESPACE_FUNCTION_TABLE, timeout=timeout)
    except Exception as e:
        logger.exception(f'Failed to export the setup hook {setup_func.__name__}.')
        raise e
    return key