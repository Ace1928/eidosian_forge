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
def check_oversized_function(pickled: bytes, name: str, obj_type: str, worker: 'ray.Worker') -> None:
    """Send a warning message if the pickled function is too large.

    Args:
        pickled: the pickled function.
        name: name of the pickled object.
        obj_type: type of the pickled object, can be 'function',
            'remote function', or 'actor'.
        worker: the worker used to send warning message. message will be logged
            locally if None.
    """
    length = len(pickled)
    if length <= ray_constants.FUNCTION_SIZE_WARN_THRESHOLD:
        return
    elif length < ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD:
        warning_message = 'The {} {} is very large ({} MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.'.format(obj_type, name, length // (1024 * 1024))
        if worker:
            push_error_to_driver(worker, ray_constants.PICKLING_LARGE_OBJECT_PUSH_ERROR, 'Warning: ' + warning_message, job_id=worker.current_job_id)
    else:
        error = 'The {} {} is too large ({} MiB > FUNCTION_SIZE_ERROR_THRESHOLD={} MiB). Check that its definition is not implicitly capturing a large array or other object in scope. Tip: use ray.put() to put large objects in the Ray object store.'.format(obj_type, name, length // (1024 * 1024), ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD // (1024 * 1024))
        raise ValueError(error)