import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
def testcode(array_analysis):
    func_ir = array_analysis.func_ir
    for call in func_ir.blocks[0].find_exprs('call'):
        callee = func_ir.get_definition(call.func)
        if getattr(callee, 'value', None) is empty:
            if getattr(call.args[0], 'name', None) == 'n':
                break
    else:
        return
    variable_A = func_ir.get_assignee(call)
    es = array_analysis.equiv_sets[0]
    self.assertTrue(es.is_equiv('n', variable_A.name))
    shared['counter'] += 1