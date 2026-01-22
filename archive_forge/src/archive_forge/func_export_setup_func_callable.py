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
def export_setup_func_callable(runtime_env: Union[Dict[str, Any], RuntimeEnv], setup_func: Callable, worker: 'ray.Worker') -> Union[Dict[str, Any], RuntimeEnv]:
    assert isinstance(setup_func, Callable)
    try:
        key = worker.function_actor_manager.export_setup_func(setup_func, timeout=get_import_export_timeout())
    except Exception as e:
        raise ray.exceptions.RuntimeEnvSetupError('Failed to export the setup function.') from e
    env_vars = runtime_env.get('env_vars', {})
    assert ray_constants.WORKER_PROCESS_SETUP_HOOK_ENV_VAR not in env_vars, f'The env var, {ray_constants.WORKER_PROCESS_SETUP_HOOK_ENV_VAR}, is not permitted because it is reserved for the internal use.'
    env_vars[ray_constants.WORKER_PROCESS_SETUP_HOOK_ENV_VAR] = _decode_function_key(key)
    runtime_env['env_vars'] = env_vars
    runtime_env['worker_process_setup_hook'] = setup_func.__name__
    return runtime_env