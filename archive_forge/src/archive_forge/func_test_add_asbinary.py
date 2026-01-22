import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_add_asbinary(self):
    tests = [(self.asbinary, self.invalid, NotImplemented), (self.asbinary, self.asbinary, NotImplemented), (self.asbinary, self.zero, self.bin), (self.asbinary, self.one, LinearExpression([self.mon_bin, 1])), (self.asbinary, self.native, LinearExpression([self.mon_bin, 5])), (self.asbinary, self.npv, LinearExpression([self.mon_bin, self.npv])), (self.asbinary, self.param, LinearExpression([self.mon_bin, 6])), (self.asbinary, self.param_mut, LinearExpression([self.mon_bin, self.param_mut])), (self.asbinary, self.var, LinearExpression([self.mon_bin, self.mon_var])), (self.asbinary, self.mon_native, LinearExpression([self.mon_bin, self.mon_native])), (self.asbinary, self.mon_param, LinearExpression([self.mon_bin, self.mon_param])), (self.asbinary, self.mon_npv, LinearExpression([self.mon_bin, self.mon_npv])), (self.asbinary, self.linear, LinearExpression(self.linear.args + [self.mon_bin])), (self.asbinary, self.sum, SumExpression(self.sum.args + [self.bin])), (self.asbinary, self.other, SumExpression([self.bin, self.other])), (self.asbinary, self.mutable_l0, self.bin), (self.asbinary, self.mutable_l1, LinearExpression([self.mon_bin, self.mon_npv])), (self.asbinary, self.mutable_l2, SumExpression(self.mutable_l2.args + [self.bin])), (self.asbinary, self.param0, self.bin), (self.asbinary, self.param1, LinearExpression([self.mon_bin, 1])), (self.asbinary, self.mutable_l3, LinearExpression([self.mon_bin, self.npv]))]
    self._run_cases(tests, operator.add)
    self._run_cases(tests, operator.iadd)