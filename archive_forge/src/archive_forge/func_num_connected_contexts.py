import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import ray._private.ray_constants as ray_constants
from ray._private.client_mode_hook import (
from ray._private.ray_logging import setup_logger
from ray.job_config import JobConfig
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def num_connected_contexts():
    """Return the number of client connections active."""
    global _lock, _all_contexts
    with _lock:
        return len(_all_contexts)