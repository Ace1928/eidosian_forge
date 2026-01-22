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
def wait_for_event(event_listener_type: EventListenerType, *args, **kwargs) -> 'DAGNode[Event]':
    if not issubclass(event_listener_type, EventListener):
        raise TypeError(f'Event listener type is {event_listener_type.__name__}, which is not a subclass of workflow.EventListener')

    @ray.remote
    def get_message(event_listener_type: EventListenerType, *args, **kwargs) -> Event:
        event_listener = event_listener_type()
        return asyncio_run(event_listener.poll_for_event(*args, **kwargs))

    @ray.remote
    def message_committed(event_listener_type: EventListenerType, event: Event) -> Event:
        event_listener = event_listener_type()
        asyncio_run(event_listener.event_checkpointed(event))
        return event
    return message_committed.bind(event_listener_type, get_message.bind(event_listener_type, *args, **kwargs))