import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_actor_id(self) -> Optional[str]:
    """Get the current actor ID in this worker.

        ID of the actor of the current process.
        This shouldn't be used in a driver process.
        The ID will be in hex format.

        Returns:
            The current actor id in hex format in this worker. None if there's no
            actor id.
        """
    if self.worker.mode != ray._private.worker.WORKER_MODE:
        logger.warning(f'This method is only available when the process is a worker. Current mode: {self.worker.mode}')
        return None
    actor_id = self.worker.actor_id
    return actor_id.hex() if not actor_id.is_nil() else None