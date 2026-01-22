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
def test_disaggregated_vars(self):
    m = models.makeTwoTermDisj_Nonlinear()
    TransformationFactory('gdp.hull').apply_to(m)
    transBlock = m._pyomo_gdp_hull_reformulation
    disjBlock = transBlock.relaxedDisjuncts
    for i in [0, 1]:
        relaxationBlock = disjBlock[i]
        x = relaxationBlock.disaggregatedVars.x
        if i == 1:
            w = relaxationBlock.disaggregatedVars.w
            y = transBlock._disaggregatedVars[0]
        elif i == 0:
            y = relaxationBlock.disaggregatedVars.y
            w = transBlock._disaggregatedVars[1]
        self.assertIs(w.ctype, Var)
        self.assertIsInstance(x, Var)
        self.assertIs(y.ctype, Var)
        self.assertIsInstance(w.domain, RealSet)
        self.assertIsInstance(x.domain, RealSet)
        self.assertIsInstance(y.domain, RealSet)
        self.assertEqual(w.lb, 0)
        self.assertEqual(w.ub, 7)
        self.assertEqual(x.lb, 0)
        self.assertEqual(x.ub, 8)
        self.assertEqual(y.lb, -10)
        self.assertEqual(y.ub, 0)