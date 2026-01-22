import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
def test_argmax(self):

    def test_impl1(A):
        return A.argmax()

    def test_impl2(A):
        return np.argmax(A)
    n = 211
    A = np.array([1.0, 0.0, 3.0, 2.0, 3.0])
    B = np.random.randint(10, size=n).astype(np.int32)
    C = np.random.ranf((n, n))
    D = np.array([1.0, 0.0, np.nan, 2.0, 3.0])
    self.check(test_impl1, A)
    self.check(test_impl1, B)
    self.check(test_impl1, C)
    self.check(test_impl1, D)
    self.check(test_impl2, A)
    self.check(test_impl2, B)
    self.check(test_impl2, C)
    self.check(test_impl2, D)
    msg = 'attempt to get argmax of an empty sequence'
    for impl in (test_impl1, test_impl2):
        pcfunc = self.compile_parallel(impl, (types.int64[:],))
        with self.assertRaises(ValueError) as e:
            pcfunc.entry_point(np.array([], dtype=np.int64))
        self.assertIn(msg, str(e.exception))
    data_gen = lambda: self.gen_linspace_variants(1)
    self.check_variants(test_impl1, data_gen)
    self.count_parfors_variants(test_impl1, data_gen)
    self.check_variants(test_impl2, data_gen)
    self.count_parfors_variants(test_impl2, data_gen)