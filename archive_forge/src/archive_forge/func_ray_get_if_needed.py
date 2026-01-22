import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def ray_get_if_needed(obj: Any) -> Any:
    """If obj is an ObjectRef, do ray.get, otherwise return obj"""
    if isinstance(obj, ray.ObjectRef):
        return ray.get(obj)
    return obj