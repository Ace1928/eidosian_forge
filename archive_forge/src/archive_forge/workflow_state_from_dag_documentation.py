from typing import Any, List, Optional
import re
import unicodedata
import ray
from ray.workflow.common import WORKFLOW_OPTIONS
from ray.dag import DAGNode, FunctionNode, InputNode
from ray.dag.input_node import InputAttributeNode, DAGInputData
from ray import cloudpickle
from ray._private import signature
from ray._private.client_mode_hook import client_mode_should_convert
from ray.workflow import serialization_context
from ray.workflow.common import (
from ray.workflow import workflow_context
from ray.workflow.workflow_state import WorkflowExecutionState, Task

    Transform a Ray DAG to a workflow. Map FunctionNode to workflow task with
    the workflow decorator.

    Args:
        dag_node: The DAG to be converted to a workflow.
        input_context: The input data that wraps varibles for the input node of the DAG.
        workflow_id: The ID of the workflow.
    