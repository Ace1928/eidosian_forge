import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
def check_log_disjunct(self, d, pts, f, substitute_var, x):
    self.assertEqual(len(d.component_map(Constraint)), 3)
    self.assertEqual(len(d.component_map(Var)), 2)
    self.assertIsInstance(d.lambdas, Var)
    self.assertEqual(len(d.lambdas), 2)
    for lamb in d.lambdas.values():
        self.assertEqual(lamb.lb, 0)
        self.assertEqual(lamb.ub, 1)
    self.assertIsInstance(d.convex_combo, Constraint)
    assertExpressionsEqual(self, d.convex_combo.expr, d.lambdas[0] + d.lambdas[1] == 1)
    self.assertIsInstance(d.set_substitute, Constraint)
    assertExpressionsEqual(self, d.set_substitute.expr, substitute_var == f(x), places=7)
    self.assertIsInstance(d.linear_combo, Constraint)
    self.assertEqual(len(d.linear_combo), 1)
    assertExpressionsEqual(self, d.linear_combo[0].expr, x == pts[0] * d.lambdas[0] + pts[1] * d.lambdas[1])