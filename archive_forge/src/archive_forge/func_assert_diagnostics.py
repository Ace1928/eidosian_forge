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
def assert_diagnostics(self, diagnostics, parfors_count=None, fusion_info=None, nested_fusion_info=None, replaced_fns=None, hoisted_allocations=None):
    if parfors_count is not None:
        self.assertEqual(parfors_count, diagnostics.count_parfors())
    if fusion_info is not None:
        self.assert_fusion_equivalence(fusion_info, diagnostics.fusion_info)
    if nested_fusion_info is not None:
        self.assert_fusion_equivalence(nested_fusion_info, diagnostics.nested_fusion_info)
    if replaced_fns is not None:
        repl = diagnostics.replaced_fns.values()
        for x in replaced_fns:
            for replaced in repl:
                if replaced[0] == x:
                    break
            else:
                msg = 'Replacement for %s was not found. Had %s' % (x, repl)
                raise AssertionError(msg)
    if hoisted_allocations is not None:
        hoisted_allocs = diagnostics.hoisted_allocations()
        self.assertEqual(hoisted_allocations, len(hoisted_allocs))
    with captured_stdout():
        for x in range(1, 5):
            diagnostics.dump(x)