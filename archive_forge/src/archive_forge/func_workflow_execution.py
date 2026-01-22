import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional
import ray
from ray._private.ray_logging import configure_log_file, get_worker_log_file_name
from ray.workflow.common import CheckpointModeType, WorkflowStatus
@contextmanager
def workflow_execution() -> None:
    """Scope for workflow task execution."""
    global _in_workflow_execution
    try:
        _in_workflow_execution = True
        yield
    finally:
        _in_workflow_execution = False