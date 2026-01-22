import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_mon_npv(self):
    tests = [(self.mon_npv, self.invalid, NotImplemented), (self.mon_npv, self.asbinary, LinearExpression([self.mon_npv, self.mon_bin])), (self.mon_npv, self.zero, self.mon_npv), (self.mon_npv, self.one, LinearExpression([self.mon_npv, 1])), (self.mon_npv, self.native, LinearExpression([self.mon_npv, 5])), (self.mon_npv, self.npv, LinearExpression([self.mon_npv, self.npv])), (self.mon_npv, self.param, LinearExpression([self.mon_npv, 6])), (self.mon_npv, self.param_mut, LinearExpression([self.mon_npv, self.param_mut])), (self.mon_npv, self.var, LinearExpression([self.mon_npv, self.mon_var])), (self.mon_npv, self.mon_native, LinearExpression([self.mon_npv, self.mon_native])), (self.mon_npv, self.mon_param, LinearExpression([self.mon_npv, self.mon_param])), (self.mon_npv, self.mon_npv, LinearExpression([self.mon_npv, self.mon_npv])), (self.mon_npv, self.linear, LinearExpression(self.linear.args + [self.mon_npv])), (self.mon_npv, self.sum, SumExpression(self.sum.args + [self.mon_npv])), (self.mon_npv, self.other, SumExpression([self.mon_npv, self.other])), (self.mon_npv, self.mutable_l0, self.mon_npv), (self.mon_npv, self.mutable_l1, LinearExpression([self.mon_npv] + self.mutable_l1.args)), (self.mon_npv, self.mutable_l2, SumExpression(self.mutable_l2.args + [self.mon_npv])), (self.mon_npv, self.param0, self.mon_npv), (self.mon_npv, self.param1, LinearExpression([self.mon_npv, 1])), (self.mon_npv, self.mutable_l3, LinearExpression([self.mon_npv, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)