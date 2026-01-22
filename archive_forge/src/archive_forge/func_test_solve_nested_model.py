from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.environ import (
from pyomo.core.expr.compare import (
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import random
from io import StringIO
import os
from os.path import abspath, dirname, join
from filecmp import cmp
@unittest.skipIf(not linear_solvers, 'No linear solver available')
def test_solve_nested_model(self):
    m = models.makeNestedDisjunctions_NestedDisjuncts()
    m.LocalVars = Suffix(direction=Suffix.LOCAL)
    m.LocalVars[m.d1] = [m.d1.binary_indicator_var, m.d1.d3.binary_indicator_var, m.d1.d4.binary_indicator_var]
    hull = TransformationFactory('gdp.hull')
    m_hull = hull.create_using(m)
    SolverFactory(linear_solvers[0]).solve(m_hull)
    self.assertEqual(value(m_hull.d1.binary_indicator_var), 0)
    self.assertEqual(value(m_hull.d2.binary_indicator_var), 1)
    self.assertEqual(value(m_hull.x), 1.1)
    TransformationFactory('gdp.bigm').apply_to(m, targets=m.d1.disj2)
    hull.apply_to(m)
    SolverFactory(linear_solvers[0]).solve(m)
    self.assertEqual(value(m.d1.binary_indicator_var), 0)
    self.assertEqual(value(m.d2.binary_indicator_var), 1)
    self.assertEqual(value(m.x), 1.1)