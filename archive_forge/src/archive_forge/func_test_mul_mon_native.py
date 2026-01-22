import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_mon_native(self):
    tests = [(self.mon_native, self.invalid, NotImplemented), (self.mon_native, self.asbinary, ProductExpression((self.mon_native, self.bin))), (self.mon_native, self.zero, 0), (self.mon_native, self.one, self.mon_native), (self.mon_native, self.native, MonomialTermExpression((15, self.mon_native.arg(1)))), (self.mon_native, self.npv, MonomialTermExpression((NPV_ProductExpression((self.mon_native.arg(0), self.npv)), self.mon_native.arg(1)))), (self.mon_native, self.param, MonomialTermExpression((18, self.mon_native.arg(1)))), (self.mon_native, self.param_mut, MonomialTermExpression((NPV_ProductExpression((self.mon_native.arg(0), self.param_mut)), self.mon_native.arg(1)))), (self.mon_native, self.var, ProductExpression((self.mon_native, self.var))), (self.mon_native, self.mon_native, ProductExpression((self.mon_native, self.mon_native))), (self.mon_native, self.mon_param, ProductExpression((self.mon_native, self.mon_param))), (self.mon_native, self.mon_npv, ProductExpression((self.mon_native, self.mon_npv))), (self.mon_native, self.linear, ProductExpression((self.mon_native, self.linear))), (self.mon_native, self.sum, ProductExpression((self.mon_native, self.sum))), (self.mon_native, self.other, ProductExpression((self.mon_native, self.other))), (self.mon_native, self.mutable_l0, 0), (self.mon_native, self.mutable_l1, ProductExpression((self.mon_native, self.mon_npv))), (self.mon_native, self.mutable_l2, ProductExpression((self.mon_native, self.mutable_l2))), (self.mon_native, self.param0, 0), (self.mon_native, self.param1, self.mon_native), (self.mon_native, self.mutable_l3, MonomialTermExpression((NPV_ProductExpression((self.mon_native.arg(0), self.npv)), self.mon_native.arg(1))))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)