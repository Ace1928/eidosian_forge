import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_mutable_l3(self):
    tests = [(self.mutable_l3, self.invalid, NotImplemented), (self.mutable_l3, self.asbinary, PowExpression((self.npv, self.bin))), (self.mutable_l3, self.zero, 1), (self.mutable_l3, self.one, self.npv), (self.mutable_l3, self.native, NPV_PowExpression((self.npv, 5))), (self.mutable_l3, self.npv, NPV_PowExpression((self.npv, self.npv))), (self.mutable_l3, self.param, NPV_PowExpression((self.npv, 6))), (self.mutable_l3, self.param_mut, NPV_PowExpression((self.npv, self.param_mut))), (self.mutable_l3, self.var, PowExpression((self.npv, self.var))), (self.mutable_l3, self.mon_native, PowExpression((self.npv, self.mon_native))), (self.mutable_l3, self.mon_param, PowExpression((self.npv, self.mon_param))), (self.mutable_l3, self.mon_npv, PowExpression((self.npv, self.mon_npv))), (self.mutable_l3, self.linear, PowExpression((self.npv, self.linear))), (self.mutable_l3, self.sum, PowExpression((self.npv, self.sum))), (self.mutable_l3, self.other, PowExpression((self.npv, self.other))), (self.mutable_l3, self.mutable_l0, 1), (self.mutable_l3, self.mutable_l1, PowExpression((self.npv, self.mon_npv))), (self.mutable_l3, self.mutable_l2, PowExpression((self.npv, self.mutable_l2))), (self.mutable_l3, self.param0, 1), (self.mutable_l3, self.param1, self.npv), (self.mutable_l3, self.mutable_l3, NPV_PowExpression((self.npv, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)