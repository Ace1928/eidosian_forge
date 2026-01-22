import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
@property
def current_actor(self):
    """Get the current actor handle of this actor itsself.

        Returns:
            The handle of current actor.
        """
    worker = self.worker
    worker.check_connected()
    actor_id = worker.actor_id
    if actor_id.is_nil():
        raise RuntimeError('This method is only available in an actor.')
    return worker.core_worker.get_actor_handle(actor_id)