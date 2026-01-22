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
def checkTransformedRule1(self, m, i):
    c = m.rule1[i]
    self.assertEqual(c.lower, 4)
    self.assertIsNone(c.upper)
    assertExpressionsEqual(self, c.body, EXPR.LinearExpression([EXPR.MonomialTermExpression((2, m.x[i])), EXPR.MonomialTermExpression((1, m._core_add_slack_variables.component('_slack_plus_rule1[%s]' % i)))]))