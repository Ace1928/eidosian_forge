import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
def test_block(self):

    def gen_block():
        parent = ir.Scope(None, self.loc1)
        tmp = ir.Block(parent, self.loc2)
        assign1 = ir.Assign(self.var_a, self.var_b, self.loc3)
        assign2 = ir.Assign(self.var_a, self.var_c, self.loc3)
        assign3 = ir.Assign(self.var_c, self.var_b, self.loc3)
        tmp.append(assign1)
        tmp.append(assign2)
        tmp.append(assign3)
        return tmp
    a = gen_block()
    b = gen_block()
    c = gen_block().append(ir.Assign(self.var_a, self.var_b, self.loc3))
    self.check(a, same=[b], different=[c])