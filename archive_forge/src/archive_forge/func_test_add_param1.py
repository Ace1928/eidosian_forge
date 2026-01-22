import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_param1(self):
    tests = [(self.param1, self.invalid, NotImplemented), (self.param1, self.asbinary, LinearExpression([1, self.mon_bin])), (self.param1, self.zero, 1), (self.param1, self.one, 2), (self.param1, self.native, 6), (self.param1, self.npv, NPV_SumExpression([1, self.npv])), (self.param1, self.param, 7), (self.param1, self.param_mut, NPV_SumExpression([1, self.param_mut])), (self.param1, self.var, LinearExpression([1, self.mon_var])), (self.param1, self.mon_native, LinearExpression([1, self.mon_native])), (self.param1, self.mon_param, LinearExpression([1, self.mon_param])), (self.param1, self.mon_npv, LinearExpression([1, self.mon_npv])), (self.param1, self.linear, LinearExpression(self.linear.args + [1])), (self.param1, self.sum, SumExpression(self.sum.args + [1])), (self.param1, self.other, SumExpression([1, self.other])), (self.param1, self.mutable_l0, 1), (self.param1, self.mutable_l1, LinearExpression([1] + self.mutable_l1.args)), (self.param1, self.mutable_l2, SumExpression(self.mutable_l2.args + [1])), (self.param1, self.param0, 1), (self.param1, self.param1, 2), (self.param1, self.mutable_l3, NPV_SumExpression([1, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)