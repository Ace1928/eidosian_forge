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
@PublicAPI(stability='alpha')
def delete_job(self, job_id: str) -> bool:
    """Delete a job in a terminal state and all of its associated data.

        If the job is not already in a terminal state, raises an error.
        This does not delete the job logs from disk.
        Submitting a job with the same submission ID as a previously
        deleted job is not supported and may lead to unexpected behavior.

        Example:
            >>> from ray.job_submission import JobSubmissionClient
            >>> client = JobSubmissionClient() # doctest: +SKIP
            >>> job_id = client.submit_job(entrypoint="echo hello") # doctest: +SKIP
            >>> client.delete_job(job_id) # doctest: +SKIP
            True

        Args:
            job_id: submission ID for the job to be deleted.

        Returns:
            True if the job was deleted, otherwise False.

        Raises:
            RuntimeError: If the job does not exist, if the request to the
                job server fails, or if the job is not in a terminal state.
        """
    logger.debug(f'Deleting job with job_id={job_id}.')
    r = self._do_request('DELETE', f'/api/jobs/{job_id}')
    if r.status_code == 200:
        return JobDeleteResponse(**r.json()).deleted
    else:
        self._raise_error(r)