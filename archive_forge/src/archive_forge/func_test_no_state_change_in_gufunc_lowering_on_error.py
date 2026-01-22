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
def test_no_state_change_in_gufunc_lowering_on_error(self):
    BROKEN_MSG = 'BROKEN_MSG'

    @register_pass(mutates_CFG=True, analysis_only=False)
    class BreakParfors(AnalysisPass):
        _name = 'break_parfors'

        def __init__(self):
            AnalysisPass.__init__(self)

        def run_pass(self, state):
            for blk in state.func_ir.blocks.values():
                for stmt in blk.body:
                    if isinstance(stmt, numba.parfors.parfor.Parfor):

                        class Broken(list):

                            def difference(self, other):
                                raise errors.LoweringError(BROKEN_MSG)
                        stmt.races = Broken()
                return True

    class BreakParforsCompiler(CompilerBase):

        def define_pipelines(self):
            pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
            pm.add_pass_after(BreakParfors, IRLegalization)
            pm.finalize()
            return [pm]

    @njit(parallel=True, pipeline_class=BreakParforsCompiler)
    def foo():
        x = 1
        for _ in prange(1):
            x += 1
        return x
    self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)
    with self.assertRaises(errors.LoweringError) as raises:
        foo()
    self.assertIn(BROKEN_MSG, str(raises.exception))
    self.assertFalse(numba.parfors.parfor.sequential_parfor_lowering)