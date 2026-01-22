import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_native(self):
    tests = [(self.native, self.invalid, self.SKIP), (self.native, self.asbinary, MonomialTermExpression((5, self.bin))), (self.native, self.zero, 0), (self.native, self.one, 5), (self.native, self.native, 25), (self.native, self.npv, NPV_ProductExpression((5, self.npv))), (self.native, self.param, 30), (self.native, self.param_mut, NPV_ProductExpression((5, self.param_mut))), (self.native, self.var, MonomialTermExpression((5, self.var))), (self.native, self.mon_native, MonomialTermExpression((15, self.mon_native.arg(1)))), (self.native, self.mon_param, MonomialTermExpression((NPV_ProductExpression((5, self.mon_param.arg(0))), self.mon_param.arg(1)))), (self.native, self.mon_npv, MonomialTermExpression((NPV_ProductExpression((5, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.native, self.linear, ProductExpression((5, self.linear))), (self.native, self.sum, ProductExpression((5, self.sum))), (self.native, self.other, ProductExpression((5, self.other))), (self.native, self.mutable_l0, 0), (self.native, self.mutable_l1, MonomialTermExpression((NPV_ProductExpression((5, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.native, self.mutable_l2, ProductExpression((5, self.mutable_l2))), (self.native, self.param0, 0), (self.native, self.param1, 5), (self.native, self.mutable_l3, NPV_ProductExpression((5, self.npv)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)