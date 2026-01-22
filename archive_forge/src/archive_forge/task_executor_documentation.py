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

        This function resolves the inputs for the code inside
        a workflow task (works on the callee side). For outputs from other
        workflows, we resolve them into object instances inplace.

        For each ObjectRef argument, the function returns both the ObjectRef
        and the object instance. If the ObjectRef is a chain of nested
        ObjectRefs, then we resolve it recursively until we get the
        object instance, and we return the *direct* ObjectRef of the
        instance. This function does not resolve ObjectRef
        inside another object (e.g. list of ObjectRefs) to give users some
        flexibility.

        Returns:
            Instances of arguments.
        