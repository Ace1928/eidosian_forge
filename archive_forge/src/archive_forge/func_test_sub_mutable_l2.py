import math
import operator
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
import pyomo.core.expr as EXPR
from pyomo.core.expr import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import NumericValue, native_types, native_numeric_types
from .test_numeric_expr_dispatcher import Base
def test_sub_mutable_l2(self):
    tests = [(self.mutable_l2, self.invalid, NotImplemented), (self.mutable_l2, self.asbinary, SumExpression(self.mutable_l2.args + [self.minus_bin])), (self.mutable_l2, self.zero, self.mutable_l2), (self.mutable_l2, self.one, SumExpression(self.mutable_l2.args + [-1])), (self.mutable_l2, self.native, SumExpression(self.mutable_l2.args + [-5])), (self.mutable_l2, self.npv, SumExpression(self.mutable_l2.args + [self.minus_npv])), (self.mutable_l2, self.param, SumExpression(self.mutable_l2.args + [-6])), (self.mutable_l2, self.param_mut, SumExpression(self.mutable_l2.args + [self.minus_param_mut])), (self.mutable_l2, self.var, SumExpression(self.mutable_l2.args + [self.minus_var])), (self.mutable_l2, self.mon_native, SumExpression(self.mutable_l2.args + [self.minus_mon_native])), (self.mutable_l2, self.mon_param, SumExpression(self.mutable_l2.args + [self.minus_mon_param])), (self.mutable_l2, self.mon_npv, SumExpression(self.mutable_l2.args + [self.minus_mon_npv])), (self.mutable_l2, self.linear, SumExpression(self.mutable_l2.args + [self.minus_linear])), (self.mutable_l2, self.sum, SumExpression(self.mutable_l2.args + [self.minus_sum])), (self.mutable_l2, self.other, SumExpression(self.mutable_l2.args + [self.minus_other])), (self.mutable_l2, self.mutable_l0, self.mutable_l2), (self.mutable_l2, self.mutable_l1, SumExpression(self.mutable_l2.args + [self.minus_mon_npv])), (self.mutable_l2, self.mutable_l2, SumExpression(self.mutable_l2.args + [self.minus_mutable_l2])), (self.mutable_l2, self.param0, self.mutable_l2), (self.mutable_l2, self.param1, SumExpression(self.mutable_l2.args + [-1])), (self.mutable_l2, self.mutable_l3, SumExpression(self.mutable_l2.args + [self.minus_npv]))]
    self._run_cases(tests, operator.sub)