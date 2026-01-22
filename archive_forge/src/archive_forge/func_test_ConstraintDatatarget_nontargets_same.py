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
def test_ConstraintDatatarget_nontargets_same(self):
    m = self.makeModel()
    TransformationFactory('core.add_slack_variables').apply_to(m, targets=[m.rule1[2]])
    self.checkUntransformedRule1(m, 1)
    self.checkUntransformedRule1(m, 3)
    self.checkRule2(m)