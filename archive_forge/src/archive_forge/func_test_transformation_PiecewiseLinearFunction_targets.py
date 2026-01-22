import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
def test_transformation_PiecewiseLinearFunction_targets(self):
    ct.check_transformation_PiecewiseLinearFunction_targets(self, 'contrib.piecewise.inner_repn_gdp')