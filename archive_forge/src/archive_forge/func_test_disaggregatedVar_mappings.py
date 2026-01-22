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
def test_disaggregatedVar_mappings(self):
    m = models.makeTwoTermDisj_Nonlinear()
    hull = TransformationFactory('gdp.hull')
    hull.apply_to(m)
    transBlock = m._pyomo_gdp_hull_reformulation
    disjBlock = transBlock.relaxedDisjuncts
    for i in [0, 1]:
        mappings = ComponentMap()
        mappings[m.x] = disjBlock[i].disaggregatedVars.x
        if i == 1:
            mappings[m.w] = disjBlock[i].disaggregatedVars.w
            mappings[m.y] = transBlock._disaggregatedVars[0]
        elif i == 0:
            mappings[m.y] = disjBlock[i].disaggregatedVars.y
            mappings[m.w] = transBlock._disaggregatedVars[1]
        for orig, disagg in mappings.items():
            self.assertIs(hull.get_src_var(disagg), orig)
            self.assertIs(hull.get_disaggregated_var(orig, m.d[i]), disagg)