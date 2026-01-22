import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)
@classmethod
def run_parfor_pre_pass(cls, test_func, args, swap_map=None):
    tp, options, diagnostics, preparfor_pass = cls._run_parfor(test_func, args, swap_map)
    return preparfor_pass