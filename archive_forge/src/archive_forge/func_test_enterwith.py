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
def test_enterwith(self):
    a = ir.EnterWith(self.var_a, 0, 1, self.loc1)
    b = ir.EnterWith(self.var_a, 0, 1, self.loc1)
    c = ir.EnterWith(self.var_a, 0, 1, self.loc2)
    d = ir.EnterWith(self.var_b, 0, 1, self.loc1)
    e = ir.EnterWith(self.var_a, 1, 1, self.loc1)
    f = ir.EnterWith(self.var_a, 0, 2, self.loc1)
    self.check(a, same=[b, c], different=[d, e, f])