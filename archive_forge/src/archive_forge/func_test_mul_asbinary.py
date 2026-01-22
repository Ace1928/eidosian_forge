import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_mul_asbinary(self):
    tests = [(self.asbinary, self.invalid, NotImplemented), (self.asbinary, self.asbinary, NotImplemented), (self.asbinary, self.zero, 0), (self.asbinary, self.one, self.bin), (self.asbinary, self.native, MonomialTermExpression((5, self.bin))), (self.asbinary, self.npv, MonomialTermExpression((self.npv, self.bin))), (self.asbinary, self.param, MonomialTermExpression((6, self.bin))), (self.asbinary, self.param_mut, MonomialTermExpression((self.param_mut, self.bin))), (self.asbinary, self.var, ProductExpression((self.bin, self.var))), (self.asbinary, self.mon_native, ProductExpression((self.bin, self.mon_native))), (self.asbinary, self.mon_param, ProductExpression((self.bin, self.mon_param))), (self.asbinary, self.mon_npv, ProductExpression((self.bin, self.mon_npv))), (self.asbinary, self.linear, ProductExpression((self.bin, self.linear))), (self.asbinary, self.sum, ProductExpression((self.bin, self.sum))), (self.asbinary, self.other, ProductExpression((self.bin, self.other))), (self.asbinary, self.mutable_l0, 0), (self.asbinary, self.mutable_l1, ProductExpression((self.bin, self.mon_npv))), (self.asbinary, self.mutable_l2, ProductExpression((self.bin, self.mutable_l2))), (self.asbinary, self.param0, 0), (self.asbinary, self.param1, self.bin), (self.asbinary, self.mutable_l3, MonomialTermExpression((self.npv, self.bin)))]
    self._run_cases(tests, operator.mul)
    self._run_cases(tests, operator.imul)