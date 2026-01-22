import asyncio
import logging
import queue
from typing import Dict, List, Set, Optional, TYPE_CHECKING
import ray
from ray.workflow import common
from ray.workflow.common import WorkflowStatus, TaskID
from ray.workflow import workflow_state_from_storage
from ray.workflow import workflow_context
from ray.workflow import workflow_storage
from ray.workflow.exceptions import (
from ray.workflow.workflow_executor import WorkflowExecutor
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.workflow_context import WorkflowTaskContext
def list_non_terminating_workflows(self) -> Dict[WorkflowStatus, List[str]]:
    """List workflows whose status are not of terminated status."""
    result = {WorkflowStatus.RUNNING: [], WorkflowStatus.PENDING: []}
    for wf in self._workflow_executors.keys():
        if wf in self._running_workflows:
            result[WorkflowStatus.RUNNING].append(wf)
        else:
            result[WorkflowStatus.PENDING].append(wf)
    return result