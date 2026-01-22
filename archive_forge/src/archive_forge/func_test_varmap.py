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
def test_varmap(self):
    a = ir.VarMap()
    a.define(self.var_a, 'foo')
    a.define(self.var_b, 'bar')
    b = ir.VarMap()
    b.define(self.var_a, 'foo')
    b.define(self.var_b, 'bar')
    c = ir.VarMap()
    c.define(self.var_a, 'foo')
    c.define(self.var_c, 'bar')
    self.check(a, same=[b], different=[c])