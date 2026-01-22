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
def get_worker_log_file_name(worker_type, job_id=None):
    if job_id is None:
        job_id = os.environ.get('RAY_JOB_ID')
    if worker_type == 'WORKER':
        if job_id is None:
            job_id = ''
        worker_name = 'worker'
    else:
        job_id = ''
        worker_name = 'io_worker'
    assert ray._private.worker._global_node is not None
    assert ray._private.worker.global_worker is not None
    filename = f'{worker_name}-{ray.get_runtime_context().get_worker_id()}-'
    if job_id:
        filename += f'{job_id}-'
    filename += f'{os.getpid()}'
    return filename