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
def test_all_iterations_reset_chunksize(self):
    """ Test that all the iterations get run if you set the
            chunksize.  Also check that the chunksize that each
            worker thread sees has been reset to 0. """

    @njit(parallel=True)
    def test_impl(cs, n):
        res = np.zeros(n)
        inner_cs = np.full(n, -13)
        with numba.parallel_chunksize(cs):
            for i in numba.prange(n):
                inner_cs[i] = numba.get_parallel_chunksize()
                res[i] = 13
        return (res, inner_cs)
    for j in [1000, 997, 943, 961]:
        for i in range(15):
            res, inner_cs = test_impl(i + 1, j)
            self.assertTrue(np.all(res == 13))
            self.assertTrue(np.all(inner_cs == 0))