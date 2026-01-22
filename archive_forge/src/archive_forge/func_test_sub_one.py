import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_one(self):
    tests = [(self.one, self.invalid, NotImplemented), (self.one, self.asbinary, LinearExpression([1, self.minus_bin])), (self.one, self.zero, 1), (self.one, self.one, 0), (self.one, self.native, -4), (self.one, self.npv, NPV_SumExpression([1, self.minus_npv])), (self.one, self.param, -5), (self.one, self.param_mut, NPV_SumExpression([1, self.minus_param_mut])), (self.one, self.var, LinearExpression([1, self.minus_var])), (self.one, self.mon_native, LinearExpression([1, self.minus_mon_native])), (self.one, self.mon_param, LinearExpression([1, self.minus_mon_param])), (self.one, self.mon_npv, LinearExpression([1, self.minus_mon_npv])), (self.one, self.linear, SumExpression([1, self.minus_linear])), (self.one, self.sum, SumExpression([1, self.minus_sum])), (self.one, self.other, SumExpression([1, self.minus_other])), (self.one, self.mutable_l0, 1), (self.one, self.mutable_l1, LinearExpression([1, self.minus_mon_npv])), (self.one, self.mutable_l2, SumExpression([1, self.minus_mutable_l2])), (self.one, self.param0, 1), (self.one, self.param1, 0), (self.one, self.mutable_l3, NPV_SumExpression([1, self.minus_npv]))]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)