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
@unittest.skipIf(not _32bit, 'Only impacts 32 bit hardware')
@needs_blas
def test_unsupported_combination_raises(self):
    """
        This test is in place until issues with the 'parallel'
        target on 32 bit hardware are fixed.
        """
    with self.assertRaises(errors.UnsupportedParforsError) as raised:

        @njit(parallel=True)
        def ddot(a, v):
            return np.dot(a, v)
        A = np.linspace(0, 1, 20).reshape(2, 10)
        v = np.linspace(2, 1, 10)
        ddot(A, v)
    msg = "The 'parallel' target is not currently supported on 32 bit hardware"
    self.assertIn(msg, str(raised.exception))