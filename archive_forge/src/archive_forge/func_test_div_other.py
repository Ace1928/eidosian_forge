import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_other(self):
    tests = [(self.other, self.invalid, NotImplemented), (self.other, self.asbinary, DivisionExpression((self.other, self.bin))), (self.other, self.zero, ZeroDivisionError), (self.other, self.one, self.other), (self.other, self.native, DivisionExpression((self.other, 5))), (self.other, self.npv, DivisionExpression((self.other, self.npv))), (self.other, self.param, DivisionExpression((self.other, 6))), (self.other, self.param_mut, DivisionExpression((self.other, self.param_mut))), (self.other, self.var, DivisionExpression((self.other, self.var))), (self.other, self.mon_native, DivisionExpression((self.other, self.mon_native))), (self.other, self.mon_param, DivisionExpression((self.other, self.mon_param))), (self.other, self.mon_npv, DivisionExpression((self.other, self.mon_npv))), (self.other, self.linear, DivisionExpression((self.other, self.linear))), (self.other, self.sum, DivisionExpression((self.other, self.sum))), (self.other, self.other, DivisionExpression((self.other, self.other))), (self.other, self.mutable_l0, ZeroDivisionError), (self.other, self.mutable_l1, DivisionExpression((self.other, self.mon_npv))), (self.other, self.mutable_l2, DivisionExpression((self.other, self.mutable_l2))), (self.other, self.param0, ZeroDivisionError), (self.other, self.param1, self.other), (self.other, self.mutable_l3, DivisionExpression((self.other, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)