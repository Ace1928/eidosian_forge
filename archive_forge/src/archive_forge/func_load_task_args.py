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
def load_task_args(self, task_id: TaskID) -> ray.ObjectRef:
    """Load the input arguments of the workflow task. This must be
        done under a serialization context, otherwise the arguments would
        not be reconstructed successfully.

        Args:
            task_id: ID of the workflow task.

        Returns:
            An object ref of the input args.
        """
    with serialization_context.workflow_args_keeping_context():
        x = self._get(self._key_task_args(task_id))
    return ray.put(x)