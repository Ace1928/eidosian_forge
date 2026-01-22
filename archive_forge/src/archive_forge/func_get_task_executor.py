import time
from dataclasses import dataclass
import logging
from typing import List, Tuple, Any, Dict, Callable, TYPE_CHECKING
import ray
from ray import ObjectRef
from ray._private import signature
from ray.dag import DAGNode
from ray.workflow import workflow_context
from ray.workflow.workflow_context import get_task_status_info
from ray.workflow import serialization_context
from ray.workflow import workflow_storage
from ray.workflow.common import (
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
def get_task_executor(task_options: 'WorkflowTaskRuntimeOptions'):
    if task_options.task_type == TaskType.FUNCTION:
        task_options.ray_options['max_retries'] = 0
        task_options.ray_options['retry_exceptions'] = False
        executor = _workflow_task_executor_remote.options(**task_options.ray_options).remote
    else:
        raise ValueError(f'Invalid task type {task_options.task_type}')
    return executor