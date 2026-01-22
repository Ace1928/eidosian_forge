import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_other(self):
    tests = [(self.other, self.invalid, NotImplemented), (self.other, self.asbinary, SumExpression([self.other, self.bin])), (self.other, self.zero, self.other), (self.other, self.one, SumExpression([self.other, 1])), (self.other, self.native, SumExpression([self.other, 5])), (self.other, self.npv, SumExpression([self.other, self.npv])), (self.other, self.param, SumExpression([self.other, 6])), (self.other, self.param_mut, SumExpression([self.other, self.param_mut])), (self.other, self.var, SumExpression([self.other, self.var])), (self.other, self.mon_native, SumExpression([self.other, self.mon_native])), (self.other, self.mon_param, SumExpression([self.other, self.mon_param])), (self.other, self.mon_npv, SumExpression([self.other, self.mon_npv])), (self.other, self.linear, SumExpression([self.other, self.linear])), (self.other, self.sum, SumExpression(self.sum.args + [self.other])), (self.other, self.other, SumExpression([self.other, self.other])), (self.other, self.mutable_l0, self.other), (self.other, self.mutable_l1, SumExpression([self.other, self.mon_npv])), (self.other, self.mutable_l2, SumExpression(self.mutable_l2.args + [self.other])), (self.other, self.param0, self.other), (self.other, self.param1, SumExpression([self.other, 1])), (self.other, self.mutable_l3, SumExpression([self.other, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)