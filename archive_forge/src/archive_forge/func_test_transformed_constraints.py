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
def test_transformed_constraints(self):
    m = self.makeModel()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    nonlin_ub_list = hull.get_transformed_constraints(m.disj1.b.any_index['nonlin-ub'])
    self.assertEqual(len(nonlin_ub_list), 1)
    cons = nonlin_ub_list[0]
    self.assertIs(cons.ctype, Constraint)
    self.assertIsNone(cons.lower)
    self.assertEqual(value(cons.upper), 0)
    repn = generate_standard_repn(cons.body)
    self.assertEqual(str(repn.nonlinear_expr), '(0.9999*disj1.binary_indicator_var + 0.0001)*(_pyomo_gdp_hull_reformulation.relaxedDisjuncts[0].disaggregatedVars.y/(0.9999*disj1.binary_indicator_var + 0.0001))**2')
    self.assertEqual(len(repn.nonlinear_vars), 2)
    self.assertIs(repn.nonlinear_vars[0], m.disj1.binary_indicator_var)
    self.assertIs(repn.nonlinear_vars[1], hull.get_disaggregated_var(m.y, m.disj1))
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], m.disj1.binary_indicator_var)
    self.assertEqual(repn.linear_coefs[0], -4)
    nonlin_lb_list = hull.get_transformed_constraints(m.disj2.non_lin_lb)
    self.assertEqual(len(nonlin_lb_list), 1)
    cons = nonlin_lb_list[0]
    self.assertIs(cons.ctype, Constraint)
    self.assertIsNone(cons.lower)
    self.assertEqual(value(cons.upper), 0)
    repn = generate_standard_repn(cons.body)
    self.assertEqual(str(repn.nonlinear_expr), '- ((0.9999*disj2.binary_indicator_var + 0.0001)*log(1 + _pyomo_gdp_hull_reformulation.relaxedDisjuncts[1].disaggregatedVars.y/(0.9999*disj2.binary_indicator_var + 0.0001)))')
    self.assertEqual(len(repn.nonlinear_vars), 2)
    self.assertIs(repn.nonlinear_vars[0], m.disj2.binary_indicator_var)
    self.assertIs(repn.nonlinear_vars[1], hull.get_disaggregated_var(m.y, m.disj2))
    self.assertEqual(repn.constant, 0)
    self.assertEqual(len(repn.linear_vars), 1)
    self.assertIs(repn.linear_vars[0], m.disj2.binary_indicator_var)
    self.assertEqual(repn.linear_coefs[0], 1)