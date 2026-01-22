import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_mutable_l1(self):
    tests = [(self.mutable_l1, self.invalid, NotImplemented), (self.mutable_l1, self.asbinary, ProductExpression((self.mon_npv, self.bin))), (self.mutable_l1, self.zero, 0), (self.mutable_l1, self.one, self.mon_npv), (self.mutable_l1, self.native, MonomialTermExpression((NPV_ProductExpression((self.mon_npv.arg(0), 5)), self.mon_npv.arg(1)))), (self.mutable_l1, self.npv, MonomialTermExpression((NPV_ProductExpression((self.mon_npv.arg(0), self.npv)), self.mon_npv.arg(1)))), (self.mutable_l1, self.param, MonomialTermExpression((NPV_ProductExpression((self.mon_npv.arg(0), 6)), self.mon_npv.arg(1)))), (self.mutable_l1, self.param_mut, MonomialTermExpression((NPV_ProductExpression((self.mon_npv.arg(0), self.param_mut)), self.mon_npv.arg(1)))), (self.mutable_l1, self.var, ProductExpression((self.mon_npv, self.var))), (self.mutable_l1, self.mon_native, ProductExpression((self.mon_npv, self.mon_native))), (self.mutable_l1, self.mon_param, ProductExpression((self.mon_npv, self.mon_param))), (self.mutable_l1, self.mon_npv, ProductExpression((self.mon_npv, self.mon_npv))), (self.mutable_l1, self.linear, ProductExpression((self.mon_npv, self.linear))), (self.mutable_l1, self.sum, ProductExpression((self.mon_npv, self.sum))), (self.mutable_l1, self.other, ProductExpression((self.mon_npv, self.other))), (self.mutable_l1, self.mutable_l0, 0), (self.mutable_l1, self.mutable_l1, ProductExpression((self.mon_npv, self.mon_npv))), (self.mutable_l1, self.mutable_l2, ProductExpression((self.mon_npv, self.mutable_l2))), (self.mutable_l1, self.param0, 0), (self.mutable_l1, self.param1, self.mon_npv), (self.mutable_l1, self.mutable_l3, MonomialTermExpression((NPV_ProductExpression((self.mon_npv.arg(0), self.npv)), self.mon_npv.arg(1))))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)