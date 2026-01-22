from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def check_ln_x_approx(self, pw, x):
    self.assertEqual(len(pw._simplices), 3)
    self.assertEqual(len(pw._linear_functions), 3)
    simplices = [(0, 1), (1, 2), (2, 3)]
    for idx, simplex in enumerate(simplices):
        self.assertEqual(pw._simplices[idx], simplices[idx])
    assertExpressionsEqual(self, pw._linear_functions[0](x), log(3) / 2 * x - log(3) / 2, places=7)
    assertExpressionsEqual(self, pw._linear_functions[1](x), log(2) / 3 * x + log(3 / 2), places=7)
    assertExpressionsEqual(self, pw._linear_functions[2](x), log(5 / 3) / 4 * x + log(6 / (5 / 3) ** (3 / 2)), places=7)