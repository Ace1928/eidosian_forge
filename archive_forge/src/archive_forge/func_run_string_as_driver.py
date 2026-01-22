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
def run_string_as_driver(driver_script: str, env: Dict=None, encode: str='utf-8'):
    """Run a driver as a separate process.

    Args:
        driver_script: A string to run as a Python script.
        env: The environment variables for the driver.

    Returns:
        The script's output.
    """
    proc = subprocess.Popen([sys.executable, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    with proc:
        output = proc.communicate(driver_script.encode(encoding=encode))[0]
        if proc.returncode:
            print(ray._private.utils.decode(output, encode_type=encode))
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output, proc.stderr)
        out = ray._private.utils.decode(output, encode_type=encode)
    return out