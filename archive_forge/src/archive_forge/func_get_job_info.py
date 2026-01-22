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
def get_job_info(self, job_id: str) -> JobDetails:
    """Get the latest status and other information associated with a job.

        Example:
            >>> from ray.job_submission import JobSubmissionClient
            >>> client = JobSubmissionClient("http://127.0.0.1:8265") # doctest: +SKIP
            >>> submission_id = client.submit_job(entrypoint="sleep 1") # doctest: +SKIP
            >>> job_submission_client.get_job_info(submission_id) # doctest: +SKIP
            JobInfo(status='SUCCEEDED', message='Job finished successfully.',
            error_type=None, start_time=1647388711, end_time=1647388712,
            metadata={}, runtime_env={})

        Args:
            job_id: The job ID or submission ID of the job whose information
                is being requested.

        Returns:
            The JobInfo for the job.

        Raises:
            RuntimeError: If the job does not exist or if the request to the
                job server fails.
        """
    r = self._do_request('GET', f'/api/jobs/{job_id}')
    if r.status_code == 200:
        return JobDetails(**r.json())
    else:
        self._raise_error(r)