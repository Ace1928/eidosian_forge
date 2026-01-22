import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_zero(self):
    tests = [(self.zero, self.invalid, NotImplemented), (self.zero, self.asbinary, self.minus_bin), (self.zero, self.zero, 0), (self.zero, self.one, -1), (self.zero, self.native, -5), (self.zero, self.npv, self.minus_npv), (self.zero, self.param, -6), (self.zero, self.param_mut, self.minus_param_mut), (self.zero, self.var, self.minus_var), (self.zero, self.mon_native, self.minus_mon_native), (self.zero, self.mon_param, self.minus_mon_param), (self.zero, self.mon_npv, self.minus_mon_npv), (self.zero, self.linear, self.minus_linear), (self.zero, self.sum, self.minus_sum), (self.zero, self.other, self.minus_other), (self.zero, self.mutable_l0, 0), (self.zero, self.mutable_l1, self.minus_mon_npv), (self.zero, self.mutable_l2, self.minus_mutable_l2), (self.zero, self.param0, 0), (self.zero, self.param1, -1), (self.zero, self.mutable_l3, self.minus_npv)]
    self._run_cases(tests, operator.sub)
    self._run_cases(tests, operator.isub)