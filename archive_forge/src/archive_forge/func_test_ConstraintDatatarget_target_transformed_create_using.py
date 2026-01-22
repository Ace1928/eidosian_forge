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
def test_ConstraintDatatarget_target_transformed_create_using(self):
    m = self.makeModel()
    m2 = TransformationFactory('core.add_slack_variables').create_using(m, targets=[m.rule1[2]])
    self.checkTransformedRule1(m2, 2)