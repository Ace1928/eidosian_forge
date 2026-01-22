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
def run_in_subprocess(code, flags=None, env=None, timeout=30):
    """Run a snippet of Python code in a subprocess with flags, if any are
    given. 'env' is passed to subprocess.Popen(). 'timeout' is passed to
    popen.communicate().

    Returns the stdout and stderr of the subprocess after its termination.
    """
    if flags is None:
        flags = []
    cmd = [sys.executable] + flags + ['-c', code]
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    out, err = popen.communicate(timeout=timeout)
    if popen.returncode != 0:
        msg = 'process failed with code %s: stderr follows\n%s\n'
        raise AssertionError(msg % (popen.returncode, err.decode()))
    return (out, err)