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
def test_transformed_constraint_nonlinear(self):
    m = models.makeTwoTermDisj_Nonlinear()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    disjBlock = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts
    disj1c = hull.get_transformed_constraints(m.d[0].c)
    self.assertEqual(len(disj1c), 1)
    cons = disj1c[0]
    self.assertIsNone(cons.lower)
    self.assertEqual(cons.upper, 0)
    repn = generate_standard_repn(cons.body)
    self.assertFalse(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 1)
    EPS_1 = 1 - EPS
    _disj = m._pyomo_gdp_hull_reformulation.relaxedDisjuncts[0]
    assertExpressionsEqual(self, cons.body, EXPR.SumExpression([EXPR.ProductExpression((EXPR.LinearExpression([EXPR.MonomialTermExpression((EPS_1, m.d[0].binary_indicator_var)), EPS]), EXPR.SumExpression([EXPR.DivisionExpression((_disj.disaggregatedVars.x, EXPR.LinearExpression([EXPR.MonomialTermExpression((EPS_1, m.d[0].binary_indicator_var)), EPS]))), EXPR.PowExpression((EXPR.DivisionExpression((_disj.disaggregatedVars.y, EXPR.LinearExpression([EXPR.MonomialTermExpression((EPS_1, m.d[0].binary_indicator_var)), EPS]))), 2))]))), EXPR.NegationExpression((EXPR.ProductExpression((0.0, EXPR.LinearExpression([1, EXPR.MonomialTermExpression((-1, m.d[0].binary_indicator_var))]))),)), EXPR.MonomialTermExpression((-14.0, m.d[0].binary_indicator_var))]))