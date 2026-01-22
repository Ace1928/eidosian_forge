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
def assert_fastmath(self, pyfunc, sig):
    """
        Asserts that the fastmath flag has some effect in that suitable
        instructions are now labelled as `fast`. Whether LLVM can actually do
        anything to optimise better now the derestrictions are supplied is
        another matter!

        Arguments:
         pyfunc - a function that contains operations with parallel semantics
         sig - the type signature of pyfunc
        """
    cres = self.compile_parallel_fastmath(pyfunc, sig)
    _ir = self._get_gufunc_ir(cres)

    def _get_fast_instructions(ir):
        splitted = ir.splitlines()
        fast_inst = []
        for x in splitted:
            m = re.search('\\bfast\\b', x)
            if m is not None:
                fast_inst.append(x)
        return fast_inst

    def _assert_fast(instrs):
        ops = ('fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp', 'call')
        for inst in instrs:
            count = 0
            for op in ops:
                match = op + ' fast'
                if match in inst:
                    count += 1
            self.assertTrue(count > 0)
    for name, guir in _ir.items():
        inst = _get_fast_instructions(guir)
        _assert_fast(inst)