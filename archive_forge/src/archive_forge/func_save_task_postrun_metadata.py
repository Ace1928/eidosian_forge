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
def save_task_postrun_metadata(self, task_id: TaskID, metadata: Dict[str, Any]):
    """Save post-run metadata of the current task.

        Args:
            task_id: ID of the workflow task.
            metadata: post-run metadata of the current task.

        Raises:
            DataSaveError: if we fail to save the post-run metadata.
        """
    self._put(self._key_task_postrun_metadata(task_id), metadata, True)