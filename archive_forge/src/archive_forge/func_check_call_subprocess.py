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
def check_call_subprocess(argv, capture_stdout=False, capture_stderr=False):
    from ray.scripts.scripts import main as ray_main
    if sys.platform == 'win32':
        result = check_call_module(ray_main, argv, capture_stdout=capture_stdout, capture_stderr=capture_stderr)
    else:
        stdout_redir = None
        stderr_redir = None
        if capture_stdout:
            stdout_redir = subprocess.PIPE
        if capture_stderr and capture_stdout:
            stderr_redir = subprocess.STDOUT
        elif capture_stderr:
            stderr_redir = subprocess.PIPE
        proc = subprocess.Popen(argv, stdout=stdout_redir, stderr=stderr_redir)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, argv, stdout, stderr)
        result = b''.join([s for s in [stdout, stderr] if s is not None])
    return result