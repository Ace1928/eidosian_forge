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
def validate_init_options(self, max_running_workflows: Optional[int], max_pending_workflows: Optional[int]):
    if max_running_workflows is not None and max_running_workflows != self._max_running_workflows or (max_pending_workflows is not None and max_pending_workflows != self._max_pending_workflows):
        raise ValueError(f'The workflow init is called again but the init optionsdoes not match the original ones. Original options: max_running_workflows={self._max_running_workflows} max_pending_workflows={self._max_pending_workflows}; New options: max_running_workflows={max_running_workflows} max_pending_workflows={max_pending_workflows}.')