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
def test_expr(self):
    a = ir.Expr('some_op', self.loc1)
    b = ir.Expr('some_op', self.loc1)
    c = ir.Expr('some_op', self.loc2)
    d = ir.Expr('some_other_op', self.loc1)
    self.check(a, same=[b, c], different=[d])