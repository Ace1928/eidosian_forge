import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import cloudpickle
from ray._private import storage
from ray.types import ObjectRef
from ray.workflow.common import (
from ray.workflow.exceptions import WorkflowNotFoundError
from ray.workflow import workflow_context
from ray.workflow import serialization
from ray.workflow import serialization_context
from ray.workflow.workflow_state import WorkflowExecutionState
from ray.workflow.storage import DataLoadError, DataSaveError, KeyNotFoundError
def save_workflow_execution_state(self, creator_task_id: TaskID, state: WorkflowExecutionState) -> None:
    """Save a workflow execution state.
        Typically, the state is translated from a Ray DAG.

        Args:
            creator_task_id: The ID of the task that creates the state.
            state: The state converted from the DAG.
        """
    assert creator_task_id != state.output_task_id
    for task_id, task in state.tasks.items():
        metadata = {**task.to_dict(), 'workflow_refs': state.upstream_dependencies[task_id]}
        self._put(self._key_task_input_metadata(task_id), metadata, True)
        self._put(self._key_task_user_metadata(task_id), task.user_metadata, True)
        workflow_id = self._workflow_id
        serialization.dump_to_storage(self._key_task_function_body(task_id), task.func_body, workflow_id, self)
        with serialization_context.workflow_args_keeping_context():
            args_obj = ray.get(state.task_input_args[task_id])
        serialization.dump_to_storage(self._key_task_args(task_id), args_obj, workflow_id, self)
    self._put(self._key_task_output_metadata(creator_task_id), {'output_task_id': state.output_task_id}, True)