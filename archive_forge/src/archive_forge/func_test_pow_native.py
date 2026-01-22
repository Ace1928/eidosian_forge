import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_native(self):
    tests = [(self.native, self.invalid, NotImplemented), (self.native, self.asbinary, PowExpression((5, self.bin))), (self.native, self.zero, 1), (self.native, self.one, 5), (self.native, self.native, 3125), (self.native, self.npv, NPV_PowExpression((5, self.npv))), (self.native, self.param, 15625), (self.native, self.param_mut, NPV_PowExpression((5, self.param_mut))), (self.native, self.var, PowExpression((5, self.var))), (self.native, self.mon_native, PowExpression((5, self.mon_native))), (self.native, self.mon_param, PowExpression((5, self.mon_param))), (self.native, self.mon_npv, PowExpression((5, self.mon_npv))), (self.native, self.linear, PowExpression((5, self.linear))), (self.native, self.sum, PowExpression((5, self.sum))), (self.native, self.other, PowExpression((5, self.other))), (self.native, self.mutable_l0, 1), (self.native, self.mutable_l1, PowExpression((5, self.mon_npv))), (self.native, self.mutable_l2, PowExpression((5, self.mutable_l2))), (self.native, self.param0, 1), (self.native, self.param1, 5), (self.native, self.mutable_l3, NPV_PowExpression((5, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)