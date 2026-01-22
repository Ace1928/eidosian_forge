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
def test_new_obj_created(self):
    m = self.makeModel()
    TransformationFactory('core.add_slack_variables').apply_to(m)
    transBlock = m.component('_core_add_slack_variables')
    obj = transBlock.component('_slack_objective')
    self.assertIsInstance(obj, Objective)
    self.assertTrue(obj.active)
    assertExpressionsEqual(self, obj.expr, EXPR.LinearExpression([EXPR.MonomialTermExpression((1, transBlock._slack_minus_rule1)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule2)), EXPR.MonomialTermExpression((1, transBlock._slack_minus_rule2)), EXPR.MonomialTermExpression((1, transBlock._slack_plus_rule3))]))