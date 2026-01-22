import cmath
import contextlib
from collections import defaultdict
import enum
import gc
import math
import platform
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import io
import ctypes
import multiprocessing as mp
import warnings
import traceback
from contextlib import contextmanager
import uuid
import importlib
import types as pytypes
from functools import cached_property
import numpy as np
from numba import testing, types
from numba.core import errors, typing, utils, config, cpu
from numba.core.typing import cffi_utils
from numba.core.compiler import (compile_extra, Flags,
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
import unittest
from numba.core.runtime import rtsys
from numba.np import numpy_support
from numba.core.runtime import _nrt_python as _nrt
from numba.core.extending import (
from numba.core.datamodel.models import OpaqueModel
def run_in_new_process_in_cache_dir(func, cache_dir, verbose=True):
    """Spawn a new process to run `func` with a temporary cache directory.

    The childprocess's stdout and stderr will be captured and redirected to
    the current process's stdout and stderr.

    Similar to ``run_in_new_process_caching()`` but the ``cache_dir`` is a
    directory path instead of a name prefix for the directory path.

    Returns
    -------
    ret : dict
        exitcode: 0 for success. 1 for exception-raised.
        stdout: str
        stderr: str
    """
    ctx = mp.get_context('spawn')
    qout = ctx.Queue()
    with override_env_config('NUMBA_CACHE_DIR', cache_dir):
        proc = ctx.Process(target=_remote_runner, args=[func, qout])
        proc.start()
        proc.join()
        stdout = qout.get_nowait()
        stderr = qout.get_nowait()
        if verbose and stdout.strip():
            print()
            print('STDOUT'.center(80, '-'))
            print(stdout)
        if verbose and stderr.strip():
            print(file=sys.stderr)
            print('STDERR'.center(80, '-'), file=sys.stderr)
            print(stderr, file=sys.stderr)
    return {'exitcode': proc.exitcode, 'stdout': stdout, 'stderr': stderr}