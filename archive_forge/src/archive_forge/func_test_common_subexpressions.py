import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def test_common_subexpressions(self, fn=neg_root_common_subexpr):
    """
        Attempt to verify that rewriting will incorporate user common
        subexpressions properly.
        """
    ns = self._test_root_function(fn)
    ir0 = ns.control_pipeline.state.func_ir.blocks
    ir1 = ns.test_pipeline.state.func_ir.blocks
    self.assertEqual(len(ir0), len(ir1))
    self.assertGreater(len(ir0[0].body), len(ir1[0].body))
    self.assertEqual(len(list(self._get_array_exprs(ir0[0].body))), 0)
    array_expr_instrs = list(self._get_array_exprs(ir1[0].body))
    self.assertGreater(len(array_expr_instrs), 1)
    array_sets = list((self._array_expr_to_set(instr.value.expr)[1] for instr in array_expr_instrs))
    for expr_set_0, expr_set_1 in zip(array_sets[:-1], array_sets[1:]):
        intersections = expr_set_0.intersection(expr_set_1)
        if intersections:
            self.fail('Common subexpressions detected in array expressions ({0})'.format(intersections))