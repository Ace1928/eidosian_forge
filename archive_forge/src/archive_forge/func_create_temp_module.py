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
@contextlib.contextmanager
def create_temp_module(source_lines, **jit_options):
    """A context manager that creates and imports a temporary module
    from sources provided in ``source_lines``.

    Optionally it is possible to provide jit options for ``jit_module`` if it
    is explicitly used in ``source_lines`` like ``jit_module({jit_options})``.
    """
    try:
        tempdir = temp_directory('test_temp_module')
        temp_module_name = 'test_temp_module_{}'.format(str(uuid.uuid4()).replace('-', '_'))
        temp_module_path = os.path.join(tempdir, temp_module_name + '.py')
        jit_options = _format_jit_options(**jit_options)
        with open(temp_module_path, 'w') as f:
            lines = source_lines.format(jit_options=jit_options)
            f.write(lines)
        sys.path.insert(0, tempdir)
        test_module = importlib.import_module(temp_module_name)
        yield test_module
    finally:
        sys.modules.pop(temp_module_name, None)
        sys.path.remove(tempdir)
        shutil.rmtree(tempdir)