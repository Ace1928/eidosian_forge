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
def configure_log_file(out_file, err_file):
    if out_file is None or err_file is None:
        return
    stdout_fileno = sys.stdout.fileno()
    stderr_fileno = sys.stderr.fileno()
    os.dup2(out_file.fileno(), stdout_fileno)
    os.dup2(err_file.fileno(), stderr_fileno)
    sys.stdout = ray._private.utils.open_log(stdout_fileno, unbuffered=True, closefd=False)
    sys.stderr = ray._private.utils.open_log(stderr_fileno, unbuffered=True, closefd=False)