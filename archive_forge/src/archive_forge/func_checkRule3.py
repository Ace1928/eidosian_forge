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
def checkRule3(self, m):
    cons = m.rule3
    transBlock = m.component('_core_add_slack_variables')
    self.assertIsNone(cons.upper)
    self.assertEqual(cons.lower, 0.1)
    assertExpressionsEqual(self, cons.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, m.x)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule3))]))