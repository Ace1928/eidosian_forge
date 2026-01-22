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
@skip_unless_scipy
def test_issue6102(self):

    @njit(parallel=True)
    def f(r):
        for ir in prange(r.shape[0]):
            dist = np.inf
            tr = np.array([0, 0, 0], dtype=np.float32)
            for i in [1, 0, -1]:
                dist_t = np.linalg.norm(r[ir, :] + i)
                if dist_t < dist:
                    dist = dist_t
                    tr = np.array([i, i, i], dtype=np.float32)
            r[ir, :] += tr
        return r
    r = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    self.assertPreciseEqual(f(r), f.py_func(r))