import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_mon_param(self):
    tests = [(self.mon_param, self.invalid, NotImplemented), (self.mon_param, self.asbinary, LinearExpression([self.mon_param, self.minus_bin])), (self.mon_param, self.zero, self.mon_param), (self.mon_param, self.one, LinearExpression([self.mon_param, -1])), (self.mon_param, self.native, LinearExpression([self.mon_param, -5])), (self.mon_param, self.npv, LinearExpression([self.mon_param, self.minus_npv])), (self.mon_param, self.param, LinearExpression([self.mon_param, -6])), (self.mon_param, self.param_mut, LinearExpression([self.mon_param, self.minus_param_mut])), (self.mon_param, self.var, LinearExpression([self.mon_param, self.minus_var])), (self.mon_param, self.mon_native, LinearExpression([self.mon_param, self.minus_mon_native])), (self.mon_param, self.mon_param, LinearExpression([self.mon_param, self.minus_mon_param])), (self.mon_param, self.mon_npv, LinearExpression([self.mon_param, self.minus_mon_npv])), (self.mon_param, self.linear, SumExpression([self.mon_param, self.minus_linear])), (self.mon_param, self.sum, SumExpression([self.mon_param, self.minus_sum])), (self.mon_param, self.other, SumExpression([self.mon_param, self.minus_other])), (self.mon_param, self.mutable_l0, self.mon_param), (self.mon_param, self.mutable_l1, LinearExpression([self.mon_param, self.minus_mon_npv])), (self.mon_param, self.mutable_l2, SumExpression([self.mon_param, self.minus_mutable_l2])), (self.mon_param, self.param0, self.mon_param), (self.mon_param, self.param1, LinearExpression([self.mon_param, -1])), (self.mon_param, self.mutable_l3, LinearExpression([self.mon_param, self.minus_npv]))]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)