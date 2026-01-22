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
def submit_job(self, *, entrypoint: str, job_id: Optional[str]=None, runtime_env: Optional[Dict[str, Any]]=None, metadata: Optional[Dict[str, str]]=None, submission_id: Optional[str]=None, entrypoint_num_cpus: Optional[Union[int, float]]=None, entrypoint_num_gpus: Optional[Union[int, float]]=None, entrypoint_memory: Optional[int]=None, entrypoint_resources: Optional[Dict[str, float]]=None) -> str:
    """Submit and execute a job asynchronously.

        When a job is submitted, it runs once to completion or failure. Retries or
        different runs with different parameters should be handled by the
        submitter. Jobs are bound to the lifetime of a Ray cluster, so if the
        cluster goes down, all running jobs on that cluster will be terminated.

        Example:
            >>> from ray.job_submission import JobSubmissionClient
            >>> client = JobSubmissionClient("http://127.0.0.1:8265") # doctest: +SKIP
            >>> client.submit_job( # doctest: +SKIP
            ...     entrypoint="python script.py",
            ...     runtime_env={
            ...         "working_dir": "./",
            ...         "pip": ["requests==2.26.0"]
            ...     }
            ... )  # doctest: +SKIP
            'raysubmit_4LamXRuQpYdSMg7J'

        Args:
            entrypoint: The shell command to run for this job.
            submission_id: A unique ID for this job.
            runtime_env: The runtime environment to install and run this job in.
            metadata: Arbitrary data to store along with this job.
            job_id: DEPRECATED. This has been renamed to submission_id
            entrypoint_num_cpus: The quantity of CPU cores to reserve for the execution
                of the entrypoint command, separately from any tasks or actors launched
                by it. Defaults to 0.
            entrypoint_num_gpus: The quantity of GPUs to reserve for the execution
                of the entrypoint command, separately from any tasks or actors launched
                by it. Defaults to 0.
            entrypoint_memory: The quantity of memory to reserve for the
                execution of the entrypoint command, separately from any tasks or
                actors launched by it. Defaults to 0.
            entrypoint_resources: The quantity of custom resources to reserve for the
                execution of the entrypoint command, separately from any tasks or
                actors launched by it.

        Returns:
            The submission ID of the submitted job.  If not specified,
            this is a randomly generated unique ID.

        Raises:
            RuntimeError: If the request to the job server fails, or if the specified
                submission_id has already been used by a job on this cluster.
        """
    if job_id:
        logger.warning('job_id kwarg is deprecated. Please use submission_id instead.')
    if entrypoint_num_cpus or entrypoint_num_gpus or entrypoint_resources:
        self._check_connection_and_version(min_version='2.2', version_error_message='`entrypoint_num_cpus`, `entrypoint_num_gpus`, and `entrypoint_resources` kwargs are not supported on the Ray cluster. Please ensure the cluster is running Ray 2.2 or higher.')
    if entrypoint_memory:
        self._check_connection_and_version(min_version='2.8', version_error_message='`entrypoint_memory` kwarg is not supported on the Ray cluster. Please ensure the cluster is running Ray 2.8 or higher.')
    runtime_env = runtime_env or {}
    metadata = metadata or {}
    metadata.update(self._default_metadata)
    self._upload_working_dir_if_needed(runtime_env)
    self._upload_py_modules_if_needed(runtime_env)
    setup_hook = runtime_env.get('worker_process_setup_hook')
    if setup_hook and (not isinstance(setup_hook, str)):
        raise ValueError(f'Invalid type {type(setup_hook)} for `worker_process_setup_hook`. When a job submission API is used, `worker_process_setup_hook` only allows a string type (module name). Specify `worker_process_setup_hook` via ray.init within a driver to use a `Callable` type. ')
    runtime_env = RuntimeEnv(**runtime_env).to_dict()
    submission_id = submission_id or job_id
    req = JobSubmitRequest(entrypoint=entrypoint, submission_id=submission_id, runtime_env=runtime_env, metadata=metadata, entrypoint_num_cpus=entrypoint_num_cpus, entrypoint_num_gpus=entrypoint_num_gpus, entrypoint_memory=entrypoint_memory, entrypoint_resources=entrypoint_resources)
    json_data = strip_keys_with_value_none(dataclasses.asdict(req))
    logger.debug(f'Submitting job with submission_id={submission_id}.')
    r = self._do_request('POST', '/api/jobs/', json_data=json_data)
    if r.status_code == 200:
        return JobSubmitResponse(**r.json()).submission_id
    else:
        self._raise_error(r)