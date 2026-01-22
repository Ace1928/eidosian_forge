import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_sum(self):
    tests = [(self.sum, self.invalid, NotImplemented), (self.sum, self.asbinary, SumExpression(self.sum.args + [self.bin])), (self.sum, self.zero, self.sum), (self.sum, self.one, SumExpression(self.sum.args + [1])), (self.sum, self.native, SumExpression(self.sum.args + [5])), (self.sum, self.npv, SumExpression(self.sum.args + [self.npv])), (self.sum, self.param, SumExpression(self.sum.args + [6])), (self.sum, self.param_mut, SumExpression(self.sum.args + [self.param_mut])), (self.sum, self.var, SumExpression(self.sum.args + [self.var])), (self.sum, self.mon_native, SumExpression(self.sum.args + [self.mon_native])), (self.sum, self.mon_param, SumExpression(self.sum.args + [self.mon_param])), (self.sum, self.mon_npv, SumExpression(self.sum.args + [self.mon_npv])), (self.sum, self.linear, SumExpression(self.sum.args + [self.linear])), (self.sum, self.sum, SumExpression(self.sum.args + self.sum.args)), (self.sum, self.other, SumExpression(self.sum.args + [self.other])), (self.sum, self.mutable_l0, self.sum), (self.sum, self.mutable_l1, SumExpression(self.sum.args + self.mutable_l1.args)), (self.sum, self.mutable_l2, SumExpression(self.sum.args + self.mutable_l2.args)), (self.sum, self.param0, self.sum), (self.sum, self.param1, SumExpression(self.sum.args + [1])), (self.sum, self.mutable_l3, SumExpression(self.sum.args + [self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)