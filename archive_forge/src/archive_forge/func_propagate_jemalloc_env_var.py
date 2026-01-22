import base64
import collections
import errno
import io
import json
import logging
import mmap
import multiprocessing
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, IO, AnyStr
import psutil
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
from ray._raylet import GcsClient, GcsClientOptions
from ray.core.generated.common_pb2 import Language
from ray._private.ray_constants import RAY_NODE_IP_FILENAME
def propagate_jemalloc_env_var(*, jemalloc_path: str, jemalloc_conf: str, jemalloc_comps: List[str], process_type: str):
    """Read the jemalloc memory profiling related
    env var and return the dictionary that translates
    them to proper jemalloc related env vars.

    For example, if users specify `RAY_JEMALLOC_LIB_PATH`,
    it is translated into `LD_PRELOAD` which is needed to
    run Jemalloc as a shared library.

    Params:
        jemalloc_path: The path to the jemalloc shared library.
        jemalloc_conf: `,` separated string of jemalloc config.
        jemalloc_comps: The list of Ray components
            that we will profile.
        process_type: The process type that needs jemalloc
            env var for memory profiling. If it doesn't match one of
            jemalloc_comps, the function will return an empty dict.

    Returns:
        dictionary of {env_var: value}
            that are needed to jemalloc profiling. The caller can
            call `dict.update(return_value_of_this_func)` to
            update the dict of env vars. If the process_type doesn't
            match jemalloc_comps, it will return an empty dict.
    """
    assert isinstance(jemalloc_comps, list)
    assert process_type is not None
    process_type = process_type.lower()
    if not jemalloc_path:
        return {}
    env_vars = {'LD_PRELOAD': jemalloc_path, 'RAY_LD_PRELOAD': '1'}
    if process_type in jemalloc_comps and jemalloc_conf:
        env_vars.update({'MALLOC_CONF': jemalloc_conf})
    return env_vars