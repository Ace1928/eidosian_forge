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
@ray.remote(num_cpus=0)
def load_task_output_from_storage(workflow_id: str, task_id: Optional[TaskID]):
    wf_store = workflow_storage.WorkflowStorage(workflow_id)
    tid = wf_store.inspect_output(task_id)
    if tid is not None:
        return wf_store.load_task_output(tid)
    if task_id is not None:
        raise ValueError(f"Cannot load output from task id '{task_id}' in workflow '{workflow_id}'")
    else:
        raise ValueError(f"Cannot load output from workflow '{workflow_id}'")