import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_one(self):
    tests = [(self.one, self.invalid, NotImplemented), (self.one, self.asbinary, PowExpression((1, self.bin))), (self.one, self.zero, 1), (self.one, self.one, 1), (self.one, self.native, 1), (self.one, self.npv, NPV_PowExpression((1, self.npv))), (self.one, self.param, 1), (self.one, self.param_mut, NPV_PowExpression((1, self.param_mut))), (self.one, self.var, PowExpression((1, self.var))), (self.one, self.mon_native, PowExpression((1, self.mon_native))), (self.one, self.mon_param, PowExpression((1, self.mon_param))), (self.one, self.mon_npv, PowExpression((1, self.mon_npv))), (self.one, self.linear, PowExpression((1, self.linear))), (self.one, self.sum, PowExpression((1, self.sum))), (self.one, self.other, PowExpression((1, self.other))), (self.one, self.mutable_l0, 1), (self.one, self.mutable_l1, PowExpression((1, self.mon_npv))), (self.one, self.mutable_l2, PowExpression((1, self.mutable_l2))), (self.one, self.param0, 1), (self.one, self.param1, 1), (self.one, self.mutable_l3, NPV_PowExpression((1, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)