import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt, compute_bounds_on_expr
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.fileutils import find_library
from pyomo.common.log import LoggingIntercept
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.core.expr.numeric_expr import (
import math
import platform
from io import StringIO
def test_encountered_bugs1(self):
    m = pyo.Block(concrete=True)
    m.x = pyo.Var(bounds=(-0.035, -0.035))
    m.y = pyo.Var(bounds=(-0.023, -0.023))
    m.c = pyo.Constraint(expr=m.x ** 2 + m.y ** 2 <= 0.0256)
    self.tightener(m)
    self.assertEqual(m.x.lb, -0.035)
    self.assertEqual(m.x.ub, -0.035)
    self.assertEqual(m.y.lb, -0.023)
    self.assertEqual(m.y.ub, -0.023)