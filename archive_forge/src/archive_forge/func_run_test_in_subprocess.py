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
def run_test_in_subprocess(maybefunc=None, timeout=60, envvars=None):
    """Runs the decorated test in a subprocess via invoking numba's test
        runner. kwargs timeout and envvars are passed through to
        subprocess_test_runner."""

    def wrapper(func):

        def inner(self, *args, **kwargs):
            if os.environ.get('SUBPROC_TEST', None) != '1':
                class_name = self.__class__.__name__
                self.subprocess_test_runner(test_module=self.__module__, test_class=class_name, test_name=func.__name__, timeout=timeout, envvars=envvars)
            else:
                func(self)
        return inner
    if isinstance(maybefunc, pytypes.FunctionType):
        return wrapper(maybefunc)
    else:
        return wrapper