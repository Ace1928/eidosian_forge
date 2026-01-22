from pyomo.common.dependencies import dill_available
import pyomo.common.unittest as unittest
from pyomo.common.deprecation import RenamedClass
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.core.base import constraint, _ConstraintData
from pyomo.core.expr.compare import (
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.common.log import LoggingIntercept
import logging
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct
import pyomo.network as ntwk
import random
from io import StringIO
def test_nonlinear_bigM(self):
    m = models.makeTwoTermDisj_Nonlinear()
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m)
    disjBlock = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts
    c = bigm.get_transformed_constraints(m.d[0].c)
    self.assertEqual(len(c), 1)
    c_ub = c[0]
    self.assertTrue(c_ub.active)
    repn = generate_standard_repn(c_ub.body)
    self.assertFalse(repn.is_linear())
    self.assertEqual(len(repn.linear_vars), 2)
    ct.check_linear_coef(self, repn, m.x, 1)
    ct.check_linear_coef(self, repn, m.d[0].indicator_var, 94)
    self.assertEqual(repn.constant, -94)
    self.assertEqual(c_ub.upper, m.d[0].c.upper)
    self.assertIsNone(c_ub.lower)