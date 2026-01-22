import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from ray.autoscaler._private import commands
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent  # noqa: F401
from ray.autoscaler._private.event_system import global_event_system  # noqa: F401
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def register_callback_handler(event_name: str, callback: Union[Callable[[Dict], None], List[Callable[[Dict], None]]]) -> None:
    """Registers a callback handler for autoscaler events.

    Args:
        event_name: Event that callback should be called on. See
            CreateClusterEvent for details on the events available to be
            registered against.
        callback: Callable object that is invoked
            when specified event occurs.
    """
    global_event_system.add_callback_handler(event_name, callback)