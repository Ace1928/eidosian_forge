import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_mutable_l0(self):
    tests = [(self.mutable_l0, self.invalid, NotImplemented), (self.mutable_l0, self.asbinary, 0), (self.mutable_l0, self.zero, 0), (self.mutable_l0, self.one, 0), (self.mutable_l0, self.native, 0), (self.mutable_l0, self.npv, 0), (self.mutable_l0, self.param, 0), (self.mutable_l0, self.param_mut, 0), (self.mutable_l0, self.var, 0), (self.mutable_l0, self.mon_native, 0), (self.mutable_l0, self.mon_param, 0), (self.mutable_l0, self.mon_npv, 0), (self.mutable_l0, self.linear, 0), (self.mutable_l0, self.sum, 0), (self.mutable_l0, self.other, 0), (self.mutable_l0, self.mutable_l0, 0), (self.mutable_l0, self.mutable_l1, 0), (self.mutable_l0, self.mutable_l2, 0), (self.mutable_l0, self.param0, 0), (self.mutable_l0, self.param1, 0), (self.mutable_l0, self.mutable_l3, 0)]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)