import asyncio
import binascii
from collections import defaultdict
import contextlib
import errno
import functools
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlencode, unquote, urlparse, parse_qsl, urlunparse
import warnings
from inspect import signature
from pathlib import Path
from subprocess import list2cmdline
from typing import (
import psutil
from google.protobuf import json_format
import ray
import ray._private.ray_constants as ray_constants
from ray.core.generated.runtime_env_common_pb2 import (
def get_used_memory():
    """Return the currently used system memory in bytes

    Returns:
        The total amount of used memory
    """
    docker_usage = None
    memory_usage_filename = '/sys/fs/cgroup/memory/memory.stat'
    memory_usage_filename_v2 = '/sys/fs/cgroup/memory.current'
    memory_stat_filename_v2 = '/sys/fs/cgroup/memory.stat'
    if os.path.exists(memory_usage_filename):
        docker_usage = get_cgroupv1_used_memory(memory_usage_filename)
    elif os.path.exists(memory_usage_filename_v2) and os.path.exists(memory_stat_filename_v2):
        docker_usage = get_cgroupv2_used_memory(memory_stat_filename_v2, memory_usage_filename_v2)
    if docker_usage is not None:
        return docker_usage
    return psutil.virtual_memory().used