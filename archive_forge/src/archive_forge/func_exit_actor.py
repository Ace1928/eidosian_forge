import inspect
import logging
import weakref
from typing import Any, Dict, List, Optional, Union
import ray._private.ray_constants as ray_constants
import ray._private.signature as signature
import ray._private.worker
import ray._raylet
from ray import ActorClassID, Language, cross_language
from ray._private import ray_option_utils
from ray._private.async_compat import is_async_func
from ray._private.auto_init_hook import wrap_auto_init
from ray._private.client_mode_hook import (
from ray._private.inspect_util import (
from ray._private.ray_option_utils import _warn_if_using_deprecated_placement_group
from ray._private.utils import get_runtime_env_info, parse_runtime_env
from ray._raylet import (
from ray.exceptions import AsyncioActorExit
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.placement_group import _configure_placement_group_based_on_context
from ray.util.scheduling_strategies import (
from ray.util.tracing.tracing_helper import (
@PublicAPI
def exit_actor():
    """Intentionally exit the current actor.

    This API can be used only inside an actor. Use ray.kill
    API if you'd like to kill an actor using actor handle.

    When the API is called, the actor raises an exception and exits.
    Any queued methods will fail. Any ``atexit``
    handlers installed in the actor will be run.

    Raises:
        TypeError: An exception is raised if this is a driver or this
            worker is not an actor.
    """
    worker = ray._private.worker.global_worker
    if worker.mode == ray.WORKER_MODE and (not worker.actor_id.is_nil()):
        if worker.core_worker.current_actor_is_asyncio():
            raise AsyncioActorExit()
        raise_sys_exit_with_custom_error_message('exit_actor() is called.')
    else:
        raise TypeError(f"exit_actor API is called on a non-actor worker, {worker.mode}. Call this API inside an actor methodsif you'd like to exit the actor gracefully.")