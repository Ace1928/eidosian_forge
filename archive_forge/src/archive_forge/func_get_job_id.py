import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
def get_job_id(self) -> str:
    """Get current job ID for this worker or driver.

        Job ID is the id of your Ray drivers that create tasks or actors.

        Returns:
            If called by a driver, this returns the job ID. If called in
            a task, return the job ID of the associated driver. The
            job ID will be hex format.

        Raises:
            AssertionError: If not called in a driver or worker. Generally,
                this means that ray.init() was not called.
        """
    assert ray.is_initialized(), 'Job ID is not available because Ray has not been initialized.'
    job_id = self.worker.current_job_id
    return job_id.hex()