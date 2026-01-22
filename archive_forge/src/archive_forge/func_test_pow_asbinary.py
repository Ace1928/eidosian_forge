import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_pow_asbinary(self):
    tests = [(self.asbinary, self.invalid, NotImplemented), (self.asbinary, self.asbinary, NotImplemented), (self.asbinary, self.zero, 1), (self.asbinary, self.one, self.bin), (self.asbinary, self.native, PowExpression((self.bin, 5))), (self.asbinary, self.npv, PowExpression((self.bin, self.npv))), (self.asbinary, self.param, PowExpression((self.bin, 6))), (self.asbinary, self.param_mut, PowExpression((self.bin, self.param_mut))), (self.asbinary, self.var, PowExpression((self.bin, self.var))), (self.asbinary, self.mon_native, PowExpression((self.bin, self.mon_native))), (self.asbinary, self.mon_param, PowExpression((self.bin, self.mon_param))), (self.asbinary, self.mon_npv, PowExpression((self.bin, self.mon_npv))), (self.asbinary, self.linear, PowExpression((self.bin, self.linear))), (self.asbinary, self.sum, PowExpression((self.bin, self.sum))), (self.asbinary, self.other, PowExpression((self.bin, self.other))), (self.asbinary, self.mutable_l0, 1), (self.asbinary, self.mutable_l1, PowExpression((self.bin, self.mon_npv))), (self.asbinary, self.mutable_l2, PowExpression((self.bin, self.mutable_l2))), (self.asbinary, self.param0, 1), (self.asbinary, self.param1, self.bin), (self.asbinary, self.mutable_l3, PowExpression((self.bin, self.npv)))]
    self._run_cases(tests, operator.pow)
    self._run_cases(tests, operator.ipow)