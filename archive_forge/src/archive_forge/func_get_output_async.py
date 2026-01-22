import functools
import logging
from typing import Dict, Set, List, Tuple, Union, Optional, Any
import time
import uuid
import ray
from ray.dag import DAGNode
from ray.dag.input_node import DAGInputData
from ray.remote_function import RemoteFunction
from ray.workflow.common import (
from ray.workflow import serialization, workflow_access, workflow_context
from ray.workflow.event_listener import EventListener, EventListenerType, TimerListener
from ray.workflow.workflow_storage import WorkflowStorage
from ray.workflow.workflow_state_from_dag import workflow_state_from_dag
from ray.util.annotations import PublicAPI
from ray._private.usage import usage_lib
@PublicAPI(stability='alpha')
@client_mode_wrap
def get_output_async(workflow_id: str, *, task_id: Optional[str]=None) -> ray.ObjectRef:
    """Get the output of a running workflow asynchronously.

    Args:
        workflow_id: The workflow to get the output of.
        task_id: If set, fetch the specific task output instead of the output
            of the workflow.

    Returns:
        An object reference that can be used to retrieve the workflow task result.
    """
    _ensure_workflow_initialized()
    try:
        workflow_manager = workflow_access.get_management_actor()
    except ValueError as e:
        raise ValueError('Failed to connect to the workflow management actor. The workflow could have already failed. You can use workflow.resume() or workflow.resume_async() to resume the workflow.') from e
    return workflow_manager.get_output.remote(workflow_id, task_id)