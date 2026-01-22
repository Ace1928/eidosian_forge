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
def test_suffix_M_onBlock(self):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    m.b.BigM = Suffix(direction=Suffix.LOCAL)
    m.b.BigM[None] = 34
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m)
    self.checkMs(m, -34, 34, 34, -3, 1.5)
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
    self.assertIs(l_src, m.b.BigM)
    self.assertIs(u_src, m.b.BigM)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertEqual(l_val, -34)
    self.assertEqual(u_val, 34)
    l_val, u_val = bigm.get_M_value(m.b.disjunct[0].c)
    self.assertEqual(l_val, -34)
    self.assertEqual(u_val, 34)
    (l_val, l_src, l_key), (u_val, u_src, u_key) = bigm.get_M_value_src(m.b.disjunct[1].c)
    self.assertIsNone(l_src)
    self.assertIs(u_src, m.b.BigM)
    self.assertIsNone(l_key)
    self.assertIsNone(u_key)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 34)
    l_val, u_val = bigm.get_M_value(m.b.disjunct[1].c)
    self.assertIsNone(l_val)
    self.assertEqual(u_val, 34)