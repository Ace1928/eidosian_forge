import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import ray
from ray._private.ray_logging import configure_log_file, get_worker_log_file_name
from ray.workflow.common import CheckpointModeType, WorkflowStatus
@contextmanager
def workflow_logging_context(job_id) -> None:
    """Initialize the workflow logging context.

    Workflow executions are running as remote functions from
    WorkflowManagementActor. Without logging redirection, workflow
    inner execution logs will be pushed to the driver that initially
    created WorkflowManagementActor rather than the driver that
    actually submits the current workflow execution.
    We use this conext manager to re-configure the log files to send
    the logs to the correct driver, and to restore the log files once
    the execution is done.

    Args:
        job_id: The ID of the job that submits the workflow execution.
    """
    node = ray._private.worker._global_node
    original_out_file, original_err_file = node.get_log_file_handles(get_worker_log_file_name('WORKER'))
    out_file, err_file = node.get_log_file_handles(get_worker_log_file_name('WORKER', job_id))
    try:
        configure_log_file(out_file, err_file)
        yield
    finally:
        configure_log_file(original_out_file, original_err_file)