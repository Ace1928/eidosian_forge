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
def test_suffix_M_constraintData_on_block(self):
    m = models.makeTwoTermDisj_IndexedConstraints()
    m.b.BigM = Suffix(direction=Suffix.LOCAL)
    m.b.BigM[None] = 30
    m.b.BigM[m.b.simpledisj1.c[1]] = 15
    TransformationFactory('gdp.bigm').apply_to(m)
    self.checkMs(m, -15, 15, -30, 30, 30, 30)