import traceback
import logging
import base64
import os
from typing import Dict, Any, Callable, Union, Optional
import ray
import ray._private.ray_constants as ray_constants
from ray._private.storage import _load_class
import ray.cloudpickle as pickle
from ray.runtime_env import RuntimeEnv
def load_and_execute_setup_hook_module(worker_process_setup_hook_key: str) -> Optional[str]:
    try:
        setup_func = _load_class(worker_process_setup_hook_key)
        setup_func()
        return None
    except Exception:
        error_message = f"Failed to execute the setup hook method, {worker_process_setup_hook_key} from ``ray.init(runtime_env={{'worker_process_setup_hook': {worker_process_setup_hook_key}}})``. Please make sure the given module exists and is available from ray workers. For more details, see the error trace below.\n{traceback.format_exc()}"
        return error_message