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
def test_target_objective(self):
    m = self.makeModel()
    TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1, m.rule3])
    self.assertFalse(m.obj.active)
    self.checkTargetsObj(m)