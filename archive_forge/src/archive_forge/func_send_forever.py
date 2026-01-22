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
def send_forever():
    while True:
        if self.stop_event.is_set():
            return
        start = time.time()
        for task in self.tasks:
            try:
                if start - task.last_call_succeeded_time >= task.interval_s:
                    if task.last_ref:
                        ready_refs, _ = ray.wait([task.last_ref], timeout=0)
                        if len(ready_refs) == 0:
                            continue
                    data = task.task_func()
                    task.last_call_succeeded_time = time.time()
                    if task.callback_func and ray.is_initialized():
                        task.last_ref = task.callback_func(data, send_timestamp=time.time())
            except Exception as e:
                logger.warning(f'MetricsPusher thread failed to run metric task: {e}')
        least_interval_s = math.inf
        for task in self.tasks:
            time_until_next_push = task.interval_s - (time.time() - task.last_call_succeeded_time)
            least_interval_s = min(least_interval_s, time_until_next_push)
        time.sleep(max(least_interval_s, 0))