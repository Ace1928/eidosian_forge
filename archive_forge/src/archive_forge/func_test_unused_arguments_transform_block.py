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
def test_unused_arguments_transform_block(self):
    m = models.makeTwoTermDisjOnBlock()
    m = models.add_disj_not_on_block(m)
    m.BigM = Suffix(direction=Suffix.LOCAL)
    m.BigM[None] = 1000000.0
    m.b.BigM = Suffix(direction=Suffix.LOCAL)
    m.b.BigM[None] = 15
    out = StringIO()
    with LoggingIntercept(out, 'pyomo.gdp.bigm'):
        TransformationFactory('gdp.bigm').apply_to(m.b, bigM={m: 100, m.b: 13, m.simpledisj2.c: 10})
    self.checkFirstDisjMs(m, -13, 13, 13)
    self.assertIn('Unused arguments in the bigM map! These arguments were not used by the transformation:', out.getvalue())
    self.assertIn('simpledisj2.c', out.getvalue())
    self.assertIn('unknown', out.getvalue())