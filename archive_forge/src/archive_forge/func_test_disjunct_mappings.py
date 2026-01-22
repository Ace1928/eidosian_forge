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
def test_disjunct_mappings(self):
    m = models.makeNestedDisjunctions()
    bigm = TransformationFactory('gdp.bigm')
    bigm.apply_to(m)
    disjunctBlocks = m._pyomo_gdp_bigm_reformulation.relaxedDisjuncts
    self.assertIs(m.disjunct[1].innerdisjunct[0].transformation_block, disjunctBlocks[2])
    self.assertIs(disjunctBlocks[2]._src_disjunct(), m.disjunct[1].innerdisjunct[0])
    self.assertIs(m.disjunct[1].innerdisjunct[1].transformation_block, disjunctBlocks[3])
    self.assertIs(disjunctBlocks[3]._src_disjunct(), m.disjunct[1].innerdisjunct[1])
    self.assertIs(m.simpledisjunct.innerdisjunct0.transformation_block, disjunctBlocks[0])
    self.assertIs(disjunctBlocks[0]._src_disjunct(), m.simpledisjunct.innerdisjunct0)
    self.assertIs(m.simpledisjunct.innerdisjunct1.transformation_block, disjunctBlocks[1])
    self.assertIs(disjunctBlocks[1]._src_disjunct(), m.simpledisjunct.innerdisjunct1)