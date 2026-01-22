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
def run_in_new_process_caching(func, cache_dir_prefix=__name__, verbose=True):
    """Spawn a new process to run `func` with a temporary cache directory.

    The childprocess's stdout and stderr will be captured and redirected to
    the current process's stdout and stderr.

    Returns
    -------
    ret : dict
        exitcode: 0 for success. 1 for exception-raised.
        stdout: str
        stderr: str
    """
    cache_dir = temp_directory(cache_dir_prefix)
    return run_in_new_process_in_cache_dir(func, cache_dir, verbose=verbose)