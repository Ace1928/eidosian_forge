import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
@property
def was_current_actor_reconstructed(self):
    """Check whether this actor has been restarted.

        Returns:
            Whether this actor has been ever restarted.
        """
    assert not self.actor_id.is_nil(), "This method should't be called inside Ray tasks."
    actor_info = ray._private.state.actors(self.actor_id.hex())
    return actor_info and actor_info['NumRestarts'] != 0