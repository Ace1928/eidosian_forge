import logging
from typing import Any, Dict, List, Optional
import ray._private.worker
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import pasre_pg_formatted_resources_to_original
from ray._raylet import TaskID
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import Deprecated, PublicAPI
@property
@Deprecated(message='Use get_job_id() instead', warning=True)
def job_id(self):
    """Get current job ID for this worker or driver.

        Job ID is the id of your Ray drivers that create tasks or actors.

        Returns:
            If called by a driver, this returns the job ID. If called in
            a task, return the job ID of the associated driver.

        """
    job_id = self.worker.current_job_id
    assert not job_id.is_nil()
    return job_id