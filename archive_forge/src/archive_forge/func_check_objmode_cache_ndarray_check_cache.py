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
@classmethod
def check_objmode_cache_ndarray_check_cache(cls):
    disp = cls.check_objmode_cache_ndarray()
    if len(disp.stats.cache_misses) != 0:
        raise AssertionError('unexpected cache miss')
    if len(disp.stats.cache_hits) <= 0:
        raise AssertionError('unexpected missing cache hit')