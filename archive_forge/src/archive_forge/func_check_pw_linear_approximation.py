from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def check_pw_linear_approximation(self, m):
    self.assertEqual(len(m.pw._simplices), 4)
    for i, simplex in enumerate(m.pw._simplices):
        for idx in simplex:
            self.assertIn(m.pw._points[idx], self.simplices[i])
    self.assertEqual(len(m.pw._linear_functions), 4)
    assertExpressionsStructurallyEqual(self, m.pw._linear_functions[0](m.x1, m.x2), 3 * m.x1 + 5 * m.x2 - 4, places=7)
    assertExpressionsStructurallyEqual(self, m.pw._linear_functions[1](m.x1, m.x2), 3 * m.x1 + 5 * m.x2 - 4, places=7)
    assertExpressionsStructurallyEqual(self, m.pw._linear_functions[2](m.x1, m.x2), 3 * m.x1 + 11 * m.x2 - 28, places=7)
    assertExpressionsStructurallyEqual(self, m.pw._linear_functions[3](m.x1, m.x2), 3 * m.x1 + 11 * m.x2 - 28, places=7)