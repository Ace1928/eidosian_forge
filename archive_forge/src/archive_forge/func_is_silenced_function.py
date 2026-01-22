import os
import re
import json
import socket
import contextlib
import functools
from lazyops.utils.helpers import is_coro_func
from lazyops.utils.logs import default_logger as logger
from typing import Optional, Dict, Any, Union, Callable, List, Tuple, TYPE_CHECKING
from aiokeydb.v2.types import BaseSettings, validator, lazyproperty, KeyDBUri
from aiokeydb.v2.types.static import TaskType
from aiokeydb.v2.serializers import SerializerType
from aiokeydb.v2.utils.queue import run_in_executor
from aiokeydb.v2.utils.cron import validate_cron_schedule
def is_silenced_function(self, name: str, stage: Optional[str]=None) -> bool:
    """
        Checks if a function is silenced
        """
    if name in self.tasks.silenced_functions:
        return True
    if stage:
        return name in self.tasks.silenced_functions_by_stage.get(stage, [])
    return False