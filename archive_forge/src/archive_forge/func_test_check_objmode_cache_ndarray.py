import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def test_check_objmode_cache_ndarray(self):
    cache_dir = temp_directory(self.__class__.__name__)
    with override_config('CACHE_DIR', cache_dir):
        run_in_new_process_in_cache_dir(self.populate_objmode_cache_ndarray_check_cache, cache_dir)
        res = run_in_new_process_in_cache_dir(self.check_objmode_cache_ndarray_check_cache, cache_dir)
    self.assertEqual(res['exitcode'], 0)