import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_div_mutable_l3(self):
    tests = [(self.mutable_l3, self.invalid, NotImplemented), (self.mutable_l3, self.asbinary, DivisionExpression((self.npv, self.bin))), (self.mutable_l3, self.zero, ZeroDivisionError), (self.mutable_l3, self.one, self.npv), (self.mutable_l3, self.native, NPV_DivisionExpression((self.npv, 5))), (self.mutable_l3, self.npv, NPV_DivisionExpression((self.npv, self.npv))), (self.mutable_l3, self.param, NPV_DivisionExpression((self.npv, 6))), (self.mutable_l3, self.param_mut, NPV_DivisionExpression((self.npv, self.param_mut))), (self.mutable_l3, self.var, DivisionExpression((self.npv, self.var))), (self.mutable_l3, self.mon_native, DivisionExpression((self.npv, self.mon_native))), (self.mutable_l3, self.mon_param, DivisionExpression((self.npv, self.mon_param))), (self.mutable_l3, self.mon_npv, DivisionExpression((self.npv, self.mon_npv))), (self.mutable_l3, self.linear, DivisionExpression((self.npv, self.linear))), (self.mutable_l3, self.sum, DivisionExpression((self.npv, self.sum))), (self.mutable_l3, self.other, DivisionExpression((self.npv, self.other))), (self.mutable_l3, self.mutable_l0, ZeroDivisionError), (self.mutable_l3, self.mutable_l1, DivisionExpression((self.npv, self.mon_npv))), (self.mutable_l3, self.mutable_l2, DivisionExpression((self.npv, self.mutable_l2))), (self.mutable_l3, self.param0, ZeroDivisionError), (self.mutable_l3, self.param1, self.npv), (self.mutable_l3, self.mutable_l3, NPV_DivisionExpression((self.npv, self.npv)))]
    self._run_cases(tests, operator.truediv)
    self._run_cases(tests, operator.itruediv)