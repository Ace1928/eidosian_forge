import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from pkg_resources import packaging
from ray.dashboard.utils import get_address_for_submission_client
from ray.dashboard.modules.job.utils import strip_keys_with_value_none
from ray.dashboard.modules.job.pydantic_models import (
from ray.dashboard.modules.job.common import (
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.runtime_env import RuntimeEnv
from ray.util.annotations import PublicAPI
@PublicAPI(stability='stable')
def get_job_status(self, job_id: str) -> JobStatus:
    """Get the most recent status of a job.

        Example:
            >>> from ray.job_submission import JobSubmissionClient
            >>> client = JobSubmissionClient("http://127.0.0.1:8265") # doctest: +SKIP
            >>> client.submit_job(entrypoint="echo hello") # doctest: +SKIP
            >>> client.get_job_status("raysubmit_4LamXRuQpYdSMg7J") # doctest: +SKIP
            'SUCCEEDED'

        Args:
            job_id: The job ID or submission ID of the job whose status is being
                requested.

        Returns:
            The JobStatus of the job.

        Raises:
            RuntimeError: If the job does not exist or if the request to the
                job server fails.
        """
    return self.get_job_info(job_id).status