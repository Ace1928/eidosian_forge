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
def get_output(workflow_id: str, *, task_id: Optional[str]=None) -> Any:
    """Get the output of a running workflow.

    Args:
        workflow_id: The workflow to get the output of.
        task_id: If set, fetch the specific task instead of the output of the
            workflow.

    Examples:
        .. testcode::

            from ray import workflow

            @ray.remote
            def start_trip():
                return 1

            trip = start_trip.options(**workflow.options(task_id="trip")).bind()
            res1 = workflow.run_async(trip, workflow_id="trip1")
            # you could "get_output()" in another machine
            res2 = workflow.get_output("trip1")
            assert ray.get(res1) == res2
            task_output = workflow.get_output_async("trip1", task_id="trip")
            assert ray.get(task_output) == ray.get(res1)

    Returns:
        The output of the workflow task.
    """
    return ray.get(get_output_async(workflow_id, task_id=task_id))