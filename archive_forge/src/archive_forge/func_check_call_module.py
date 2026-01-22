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
def check_call_module(main, argv, capture_stdout=False, capture_stderr=False):
    stream = io.TextIOWrapper(io.BytesIO(), encoding=sys.stdout.encoding)
    old_argv = sys.argv[:]
    try:
        sys.argv = argv[:]
        try:
            with redirect_stderr(stream if capture_stderr else sys.stderr):
                with redirect_stdout(stream if capture_stdout else sys.stdout):
                    main()
        finally:
            stream.flush()
    except SystemExit as ex:
        if ex.code:
            output = stream.buffer.getvalue()
            raise subprocess.CalledProcessError(ex.code, argv, output)
    except Exception as ex:
        output = stream.buffer.getvalue()
        raise subprocess.CalledProcessError(1, argv, output, ex.args[0])
    finally:
        sys.argv = old_argv
        if capture_stdout:
            sys.stdout.buffer.write(stream.buffer.getvalue())
        elif capture_stderr:
            sys.stderr.buffer.write(stream.buffer.getvalue())
    return stream.buffer.getvalue()