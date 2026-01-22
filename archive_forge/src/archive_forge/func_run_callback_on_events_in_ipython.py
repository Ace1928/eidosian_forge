import colorama
from dataclasses import dataclass
import logging
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Set, Tuple, Any, Optional
import ray
from ray.experimental.tqdm_ray import RAY_TQDM_MAGIC
from ray._private.ray_constants import (
from ray.util.debug import log_once
def run_callback_on_events_in_ipython(event: str, cb: Callable):
    """
    Register a callback to be run after each cell completes in IPython.
    E.g.:
        This is used to flush the logs after each cell completes.

    If IPython is not installed, this function does nothing.

    Args:
        cb: The callback to run.
    """
    if 'IPython' in sys.modules:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.events.register(event, cb)