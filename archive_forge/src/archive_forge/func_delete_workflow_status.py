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
def delete_workflow_status(self, workflow_id: str):
    """Delete status indexing for the workflow."""
    for status in WorkflowStatus:
        self._storage.delete(self._key_workflow_with_status(workflow_id, status))
    self._storage.delete(self._key_workflow_status_dirty(workflow_id))