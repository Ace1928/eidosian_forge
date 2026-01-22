import os
from os.path import abspath, dirname
from io import StringIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
import random
from pyomo.opt import check_available_solvers
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.compare import assertExpressionsEqual
def test_both_bounds_constraint_modified(self):
    m = self.makeModel()
    TransformationFactory('core.add_slack_variables').apply_to(m)
    cons = m.rule2
    transBlock = m.component('_core_add_slack_variables')
    self.assertEqual(cons.lower, 1)
    self.assertEqual(cons.upper, 3)
    assertExpressionsEqual(self, cons.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.y)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule2)), EXPR.MonomialTermExpression((-1, transBlock._slack_minus_rule2))]))