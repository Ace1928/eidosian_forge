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
def test_disjunct_M_arg(self):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    bigm = TransformationFactory('gdp.bigm')
    bigms = {m.b: 100, m.b.disjunct[1]: 13}
    bigm.apply_to(m, bigM=bigms)
    self.checkMs(m, -100, 100, 13, -3, 1.5)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisj.c)
    self.assertIsNone(l_src)
    self.assertIsNone(u_src)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertEqual(l_val, -3)
    self.assertIsNone(u_val)
    l_val, u_val = bigm.get_M_value(m.simpledisj.c)
    self.assertEqual(l_val, -3)
    self.assertIsNone(u_val)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.simpledisj2.c)
    self.assertIsNone(l_src)
    self.assertIsNone(u_src)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 1.5)
    l_val, u_val = bigm.get_M_value(m.simpledisj2.c)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 1.5)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.b.disjunct[0].c)
    self.assertIs(l_src, bigms)
    self.assertIs(u_src, bigms)
    self.assertIs(l_key, m.b)
    self.assertIs(u_key, m.b)
    self.assertEqual(l_val, -100)
    self.assertEqual(u_val, 100)
    l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
    self.assertEqual(l_val, -100)
    self.assertEqual(u_val, 100)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.b.disjunct[1].c)
    self.assertIsNone(l_src)
    self.assertIs(u_src, bigms)
    self.assertIsNone(l_key)
    self.assertIs(u_key, m.b.disjunct[1])
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 13)
    l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 13)