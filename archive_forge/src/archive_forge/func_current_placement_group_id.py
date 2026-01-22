import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
@property
@Deprecated(message='Use get_placement_group_id() instead', warning=True)
def current_placement_group_id(self):
    """Get the current Placement group ID of this worker.

        Returns:
            The current placement group id of this worker.
        """
    return self.worker.placement_group_id