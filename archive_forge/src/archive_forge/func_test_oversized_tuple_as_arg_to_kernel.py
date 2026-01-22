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
def test_oversized_tuple_as_arg_to_kernel(self):

    @njit(parallel=True)
    def oversize_tuple(idx):
        big_tup = (1, 2, 3, 4)
        z = 0
        for x in prange(10):
            z += big_tup[idx]
        return z
    with override_env_config('NUMBA_PARFOR_MAX_TUPLE_SIZE', '3'):
        with self.assertRaises(errors.UnsupportedParforsError) as raises:
            oversize_tuple(0)
    errstr = str(raises.exception)
    self.assertIn('Use of a tuple', errstr)
    self.assertIn('in a parallel region', errstr)