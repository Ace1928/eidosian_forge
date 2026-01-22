from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def test_indexed_pw_linear_function_approximate_over_simplices(self):
    m = self.make_ln_x_model()
    m.z = Var([1, 2], bounds=(-10, 10))

    def g1(x):
        return x ** 2

    def g2(x):
        return log(x)
    m.funcs = {1: g1, 2: g2}
    simplices = [(1, 3), (3, 6), (6, 10)]
    m.pw = PiecewiseLinearFunction([1, 2], simplices=simplices, function_rule=lambda m, i: m.funcs[i])
    self.check_ln_x_approx(m.pw[2], m.z[2])
    self.check_x_squared_approx(m.pw[1], m.z[1])