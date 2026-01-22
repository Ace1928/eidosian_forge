import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_assigned_resources(self):
    """Get the assigned resources to this worker.

        By default for tasks, this will return {"CPU": 1}.
        By default for actors, this will return {}. This is because
        actors do not have CPUs assigned to them by default.

        Returns:
            A dictionary mapping the name of a resource to a float, where
            the float represents the amount of that resource reserved
            for this worker.
        """
    assert self.worker.mode == ray._private.worker.WORKER_MODE, f'This method is only available when the process is a                 worker. Current mode: {self.worker.mode}'
    self.worker.check_connected()
    resource_id_map = self.worker.core_worker.resource_ids()
    resource_map = {res: sum((amt for _, amt in mapping)) for res, mapping in resource_id_map.items()}
    return pasre_pg_formatted_resources_to_original(resource_map)