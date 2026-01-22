import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_param(self):
    tests = [(self.param, self.invalid, NotImplemented), (self.param, self.asbinary, MonomialTermExpression((6, self.bin))), (self.param, self.zero, 0), (self.param, self.one, 6), (self.param, self.native, 30), (self.param, self.npv, NPV_ProductExpression((6, self.npv))), (self.param, self.param, 36), (self.param, self.param_mut, NPV_ProductExpression((6, self.param_mut))), (self.param, self.var, MonomialTermExpression((6, self.var))), (self.param, self.mon_native, MonomialTermExpression((18, self.mon_native.arg(1)))), (self.param, self.mon_param, MonomialTermExpression((NPV_ProductExpression((6, self.mon_param.arg(0))), self.mon_param.arg(1)))), (self.param, self.mon_npv, MonomialTermExpression((NPV_ProductExpression((6, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.param, self.linear, ProductExpression((6, self.linear))), (self.param, self.sum, ProductExpression((6, self.sum))), (self.param, self.other, ProductExpression((6, self.other))), (self.param, self.mutable_l0, 0), (self.param, self.mutable_l1, MonomialTermExpression((NPV_ProductExpression((6, self.mon_npv.arg(0))), self.mon_npv.arg(1)))), (self.param, self.mutable_l2, ProductExpression((6, self.mutable_l2))), (self.param, self.param0, 0), (self.param, self.param1, 6), (self.param, self.mutable_l3, NPV_ProductExpression((6, self.npv)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)