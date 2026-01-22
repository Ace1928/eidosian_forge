import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_sum(self):
    tests = [(self.sum, self.invalid, NotImplemented), (self.sum, self.asbinary, DivisionExpression((self.sum, self.bin))), (self.sum, self.zero, ZeroDivisionError), (self.sum, self.one, self.sum), (self.sum, self.native, DivisionExpression((self.sum, 5))), (self.sum, self.npv, DivisionExpression((self.sum, self.npv))), (self.sum, self.param, DivisionExpression((self.sum, 6))), (self.sum, self.param_mut, DivisionExpression((self.sum, self.param_mut))), (self.sum, self.var, DivisionExpression((self.sum, self.var))), (self.sum, self.mon_native, DivisionExpression((self.sum, self.mon_native))), (self.sum, self.mon_param, DivisionExpression((self.sum, self.mon_param))), (self.sum, self.mon_npv, DivisionExpression((self.sum, self.mon_npv))), (self.sum, self.linear, DivisionExpression((self.sum, self.linear))), (self.sum, self.sum, DivisionExpression((self.sum, self.sum))), (self.sum, self.other, DivisionExpression((self.sum, self.other))), (self.sum, self.mutable_l0, ZeroDivisionError), (self.sum, self.mutable_l1, DivisionExpression((self.sum, self.mon_npv))), (self.sum, self.mutable_l2, DivisionExpression((self.sum, self.mutable_l2))), (self.sum, self.param0, ZeroDivisionError), (self.sum, self.param1, self.sum), (self.sum, self.mutable_l3, DivisionExpression((self.sum, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)