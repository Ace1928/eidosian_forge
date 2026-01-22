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
def save_task_output(self, task_id: TaskID, ret: Any, *, exception: Optional[Exception]) -> None:
    """When a workflow task returns,
        1. If the returned object is a workflow, this means we are a nested
           workflow. We save the output metadata that points to the workflow.
        2. Otherwise, checkpoint the output.

        Args:
            task_id: The ID of the workflow task. If it is an empty string,
                it means we are in the workflow job driver process.
            ret: The returned object from a workflow task.
            exception: This task should throw exception.
        """
    if exception is None:
        ret = ray.get(ret) if isinstance(ret, ray.ObjectRef) else ret
        serialization.dump_to_storage(self._key_task_output(task_id), ret, self._workflow_id, storage=self)
    else:
        assert ret is None
        serialization.dump_to_storage(self._key_task_exception(task_id), exception, self._workflow_id, storage=self)