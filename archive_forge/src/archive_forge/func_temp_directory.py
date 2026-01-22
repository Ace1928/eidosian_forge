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
def temp_directory(prefix):
    """
    Create a temporary directory with the given *prefix* that will survive
    at least as long as this process invocation.  The temporary directory
    will be eventually deleted when it becomes stale enough.

    This is necessary because a DLL file can't be deleted while in use
    under Windows.

    An interesting side-effect is to be able to inspect the test files
    shortly after a test suite run.
    """
    _create_trashcan_dir()
    return _create_trashcan_subdir(prefix)