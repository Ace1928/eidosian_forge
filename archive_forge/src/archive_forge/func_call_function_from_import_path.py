import asyncio
import copy
import importlib
import inspect
import logging
import math
import os
import random
import string
import threading
import time
import traceback
from abc import ABC, abstractmethod
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import requests
import ray
import ray.util.serialization_addons
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.utils import import_attr
from ray._private.worker import LOCAL_MODE, SCRIPT_MODE
from ray._raylet import MessagePackSerializer
from ray.actor import ActorHandle
from ray.exceptions import RayTaskError
from ray.serve._private.constants import HTTP_PROXY_TIMEOUT, SERVE_LOGGER_NAME
from ray.types import ObjectRef
from ray.util.serialization import StandaloneSerializationContext
def call_function_from_import_path(import_path: str) -> Any:
    """Call the function given import path.

    Args:
        import_path: The import path of the function to call.
    Raises:
        ValueError: If the import path is invalid.
        TypeError: If the import path is not callable.
        RuntimeError: if the function raise exeception during execution.
    Returns:
        The result of the function call.
    """
    try:
        callback_func = import_attr(import_path)
    except Exception as e:
        raise ValueError(f'The import path {import_path} cannot be imported: {e}')
    if not callable(callback_func):
        raise TypeError(f'The import path {import_path} is not callable.')
    try:
        return callback_func()
    except Exception as e:
        raise RuntimeError(f'The function {import_path} raised an exception: {e}')