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
def test_encountered_bugs2(self):
    m = pyo.Block(concrete=True)
    m.x = pyo.Var(within=pyo.Integers)
    m.y = pyo.Var(within=pyo.Integers)
    m.c = pyo.Constraint(expr=m.x + m.y == 1)
    self.tightener(m)
    self.assertEqual(m.x.lb, None)
    self.assertEqual(m.x.ub, None)
    self.assertEqual(m.y.lb, None)
    self.assertEqual(m.y.ub, None)