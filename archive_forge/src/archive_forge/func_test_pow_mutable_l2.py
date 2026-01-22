import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_mutable_l2(self):
    tests = [(self.mutable_l2, self.invalid, NotImplemented), (self.mutable_l2, self.asbinary, PowExpression((self.mutable_l2, self.bin))), (self.mutable_l2, self.zero, 1), (self.mutable_l2, self.one, self.mutable_l2), (self.mutable_l2, self.native, PowExpression((self.mutable_l2, 5))), (self.mutable_l2, self.npv, PowExpression((self.mutable_l2, self.npv))), (self.mutable_l2, self.param, PowExpression((self.mutable_l2, 6))), (self.mutable_l2, self.param_mut, PowExpression((self.mutable_l2, self.param_mut))), (self.mutable_l2, self.var, PowExpression((self.mutable_l2, self.var))), (self.mutable_l2, self.mon_native, PowExpression((self.mutable_l2, self.mon_native))), (self.mutable_l2, self.mon_param, PowExpression((self.mutable_l2, self.mon_param))), (self.mutable_l2, self.mon_npv, PowExpression((self.mutable_l2, self.mon_npv))), (self.mutable_l2, self.linear, PowExpression((self.mutable_l2, self.linear))), (self.mutable_l2, self.sum, PowExpression((self.mutable_l2, self.sum))), (self.mutable_l2, self.other, PowExpression((self.mutable_l2, self.other))), (self.mutable_l2, self.mutable_l0, 1), (self.mutable_l2, self.mutable_l1, PowExpression((self.mutable_l2, self.mon_npv))), (self.mutable_l2, self.mutable_l2, PowExpression((self.mutable_l2, self.mutable_l2))), (self.mutable_l2, self.param0, 1), (self.mutable_l2, self.param1, self.mutable_l2), (self.mutable_l2, self.mutable_l3, PowExpression((self.mutable_l2, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)