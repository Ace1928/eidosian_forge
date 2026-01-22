import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_npv(self):
    tests = [(self.npv, self.invalid, NotImplemented), (self.npv, self.asbinary, LinearExpression([self.npv, self.mon_bin])), (self.npv, self.zero, self.npv), (self.npv, self.one, NPV_SumExpression([self.npv, 1])), (self.npv, self.native, NPV_SumExpression([self.npv, 5])), (self.npv, self.npv, NPV_SumExpression([self.npv, self.npv])), (self.npv, self.param, NPV_SumExpression([self.npv, 6])), (self.npv, self.param_mut, NPV_SumExpression([self.npv, self.param_mut])), (self.npv, self.var, LinearExpression([self.npv, self.mon_var])), (self.npv, self.mon_native, LinearExpression([self.npv, self.mon_native])), (self.npv, self.mon_param, LinearExpression([self.npv, self.mon_param])), (self.npv, self.mon_npv, LinearExpression([self.npv, self.mon_npv])), (self.npv, self.linear, LinearExpression(self.linear.args + [self.npv])), (self.npv, self.sum, SumExpression(self.sum.args + [self.npv])), (self.npv, self.other, SumExpression([self.npv, self.other])), (self.npv, self.mutable_l0, self.npv), (self.npv, self.mutable_l1, LinearExpression([self.npv] + self.mutable_l1.args)), (self.npv, self.mutable_l2, SumExpression(self.mutable_l2.args + [self.npv])), (self.npv, self.param0, self.npv), (self.npv, self.param1, NPV_SumExpression([self.npv, 1])), (self.npv, self.mutable_l3, NPV_SumExpression([self.npv, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)