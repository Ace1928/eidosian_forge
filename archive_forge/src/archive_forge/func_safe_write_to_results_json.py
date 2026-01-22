import asyncio
from datetime import datetime
import inspect
import fnmatch
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import socket
import subprocess
import sys
import tempfile
import time
import timeit
import traceback
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Dict, List, Optional
import uuid
from dataclasses import dataclass
import requests
from ray._raylet import Config
import psutil  # We must import psutil after ray because we bundle it with ray.
from ray._private import (
from ray._private.worker import RayContext
import yaml
import ray
import ray._private.gcs_utils as gcs_utils
import ray._private.memory_monitor as memory_monitor
import ray._private.services
import ray._private.utils
from ray._private.internal_api import memory_summary
from ray._private.tls_utils import generate_self_signed_tls_certs
from ray._raylet import GcsClientOptions, GlobalStateAccessor
from ray.core.generated import (
from ray.util.queue import Empty, Queue, _QueueActor
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def safe_write_to_results_json(result: dict, default_file_name: str='/tmp/release_test_output.json', env_var: Optional[str]='TEST_OUTPUT_JSON'):
    """
    Safe (atomic) write to file to guard against malforming the json
    if the job gets interrupted in the middle of writing.
    """
    test_output_json = os.environ.get(env_var, default_file_name)
    test_output_json_tmp = test_output_json + '.tmp'
    with open(test_output_json_tmp, 'wt') as f:
        json.dump(result, f)
    os.replace(test_output_json_tmp, test_output_json)
    logger.info(f'Wrote results to {test_output_json}')
    logger.info(json.dumps(result))