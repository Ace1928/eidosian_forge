import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_mutable_l1(self):
    tests = [(self.mutable_l1, self.invalid, NotImplemented), (self.mutable_l1, self.asbinary, PowExpression((self.mon_npv, self.bin))), (self.mutable_l1, self.zero, 1), (self.mutable_l1, self.one, self.mon_npv), (self.mutable_l1, self.native, PowExpression((self.mon_npv, 5))), (self.mutable_l1, self.npv, PowExpression((self.mon_npv, self.npv))), (self.mutable_l1, self.param, PowExpression((self.mon_npv, 6))), (self.mutable_l1, self.param_mut, PowExpression((self.mon_npv, self.param_mut))), (self.mutable_l1, self.var, PowExpression((self.mon_npv, self.var))), (self.mutable_l1, self.mon_native, PowExpression((self.mon_npv, self.mon_native))), (self.mutable_l1, self.mon_param, PowExpression((self.mon_npv, self.mon_param))), (self.mutable_l1, self.mon_npv, PowExpression((self.mon_npv, self.mon_npv))), (self.mutable_l1, self.linear, PowExpression((self.mon_npv, self.linear))), (self.mutable_l1, self.sum, PowExpression((self.mon_npv, self.sum))), (self.mutable_l1, self.other, PowExpression((self.mon_npv, self.other))), (self.mutable_l1, self.mutable_l0, 1), (self.mutable_l1, self.mutable_l1, PowExpression((self.mon_npv, self.mon_npv))), (self.mutable_l1, self.mutable_l2, PowExpression((self.mon_npv, self.mutable_l2))), (self.mutable_l1, self.param0, 1), (self.mutable_l1, self.param1, self.mon_npv), (self.mutable_l1, self.mutable_l3, PowExpression((self.mon_npv, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)