import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_worker_id(self) -> str:
    """Get current worker ID for this worker or driver process.

        Returns:
            A worker id in hex format for this worker or driver process.
        """
    assert ray.is_initialized(), 'Worker ID is not available because Ray has not been initialized.'
    return self.worker.worker_id.hex()