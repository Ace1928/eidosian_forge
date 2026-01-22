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
def test_disaggregate_fixed_variables(self):
    m = models.makeTwoTermDisj()
    m.x.fix(6)
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    transBlock = m.d[1].transformation_block
    self.assertIsInstance(transBlock.disaggregatedVars.component('x'), Var)
    self.assertIs(hull.get_disaggregated_var(m.x, m.d[1]), transBlock.disaggregatedVars.x)
    self.assertIs(hull.get_src_var(transBlock.disaggregatedVars.x), m.x)