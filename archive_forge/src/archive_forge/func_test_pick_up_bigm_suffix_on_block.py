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
def test_pick_up_bigm_suffix_on_block(self):
    m = models.makeTwoTermDisj_BlockOnDisj()
    m.evil[1].b.BigM = Suffix(direction=Suffix.LOCAL)
    m.evil[1].b.BigM[m.evil[1].b.c] = 2000
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m)
    cons_list = bigm.get_transformed_constraints(m.evil[1].b.c)
    ub = cons_list[1]
    self.assertEqual(ub.upper, 0)
    self.assertIsNone(ub.lower)
    repn = generate_standard_repn(ub.body)
    self.assertTrue(repn.is_linear())
    self.assertEqual(repn.constant, -2000)
    self.assertEqual(len(repn.linear_vars), 2)
    self.assertIs(repn.linear_vars[0], m.x)
    self.assertEqual(repn.linear_coefs[0], 1)
    self.assertIs(repn.linear_vars[1], m.evil[1].binary_indicator_var)
    self.assertEqual(repn.linear_coefs[1], 2000)