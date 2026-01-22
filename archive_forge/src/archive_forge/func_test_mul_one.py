import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_one(self):
    tests = [(self.one, self.invalid, self.SKIP), (self.one, self.asbinary, self.bin), (self.one, self.zero, 0), (self.one, self.one, 1), (self.one, self.native, 5), (self.one, self.npv, self.npv), (self.one, self.param, self.param), (self.one, self.param_mut, self.param_mut), (self.one, self.var, self.var), (self.one, self.mon_native, self.mon_native), (self.one, self.mon_param, self.mon_param), (self.one, self.mon_npv, self.mon_npv), (self.one, self.linear, self.linear), (self.one, self.sum, self.sum), (self.one, self.other, self.other), (self.one, self.mutable_l0, 0), (self.one, self.mutable_l1, self.mon_npv), (self.one, self.mutable_l2, self.mutable_l2), (self.one, self.param0, self.param0), (self.one, self.param1, self.param1), (self.one, self.mutable_l3, self.npv)]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)