import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_one(self):
    tests = [(self.one, self.invalid, NotImplemented), (self.one, self.asbinary, DivisionExpression((1, self.bin))), (self.one, self.zero, ZeroDivisionError), (self.one, self.one, 1.0), (self.one, self.native, 0.2), (self.one, self.npv, NPV_DivisionExpression((1, self.npv))), (self.one, self.param, 1 / 6), (self.one, self.param_mut, NPV_DivisionExpression((1, self.param_mut))), (self.one, self.var, DivisionExpression((1, self.var))), (self.one, self.mon_native, DivisionExpression((1, self.mon_native))), (self.one, self.mon_param, DivisionExpression((1, self.mon_param))), (self.one, self.mon_npv, DivisionExpression((1, self.mon_npv))), (self.one, self.linear, DivisionExpression((1, self.linear))), (self.one, self.sum, DivisionExpression((1, self.sum))), (self.one, self.other, DivisionExpression((1, self.other))), (self.one, self.mutable_l0, ZeroDivisionError), (self.one, self.mutable_l1, DivisionExpression((1, self.mon_npv))), (self.one, self.mutable_l2, DivisionExpression((1, self.mutable_l2))), (self.one, self.param0, ZeroDivisionError), (self.one, self.param1, 1.0), (self.one, self.mutable_l3, NPV_DivisionExpression((1, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)