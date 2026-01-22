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
def update_continuation_output_link(self, continuation_root_id: TaskID, latest_continuation_task_id: TaskID) -> None:
    """Update the link of the continuation output. The link points
        to the ID of the latest finished continuation task.

        Args:
            continuation_root_id: The ID of the task that returns all later
                continuations.
            latest_continuation_task_id: The ID of the latest finished
                continuation task.
        """
    try:
        metadata = self._get(self._key_task_output_metadata(continuation_root_id), True)
    except KeyNotFoundError:
        metadata = {}
    if latest_continuation_task_id != metadata.get('output_task_id') and latest_continuation_task_id != metadata.get('dynamic_output_task_id'):
        metadata['dynamic_output_task_id'] = latest_continuation_task_id
        self._put(self._key_task_output_metadata(continuation_root_id), metadata, True)