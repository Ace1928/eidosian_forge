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
def test_argument_alias_recarray_field(self):

    def test_impl(n):
        for i in range(len(n)):
            n.x[i] = 7.0
        return n
    X1 = np.zeros(10, dtype=[('x', float), ('y', int)])
    X2 = np.zeros(10, dtype=[('x', float), ('y', int)])
    X3 = np.zeros(10, dtype=[('x', float), ('y', int)])
    v1 = X1.view(np.recarray)
    v2 = X2.view(np.recarray)
    v3 = X3.view(np.recarray)
    python_res = list(test_impl(v1))
    njit_res = list(njit(test_impl)(v2))
    pa_func = njit(test_impl, parallel=True)
    pa_res = list(pa_func(v3))
    self.assertEqual(python_res, njit_res)
    self.assertEqual(python_res, pa_res)