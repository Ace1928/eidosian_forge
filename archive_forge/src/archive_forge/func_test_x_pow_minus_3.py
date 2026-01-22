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
def test_x_pow_minus_3(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.c = pyo.Constraint(expr=m.x ** (-3) == m.y)
    self.tightener(m)
    self.assertEqual(m.x.lb, None)
    self.assertEqual(m.x.ub, None)
    self.assertEqual(m.y.lb, None)
    self.assertEqual(m.y.ub, None)
    m.y.setlb(-1)
    m.y.setub(-0.125)
    self.tightener(m)
    self.assertAlmostEqual(m.x.lb, -2)
    self.assertAlmostEqual(m.x.ub, -1)
    m.x.setlb(None)
    m.x.setub(None)
    m.y.setub(0)
    self.tightener(m)
    self.assertEqual(m.x.lb, None)
    self.assertAlmostEqual(m.x.ub, -1)
    m.x.setlb(None)
    m.x.setub(None)
    m.y.setlb(-1)
    m.y.setub(1)
    self.tightener(m)
    self.assertEqual(m.x.lb, None)
    self.assertEqual(m.x.ub, None)
    m.y.setlb(0.125)
    m.y.setub(1)
    self.tightener(m)
    self.assertAlmostEqual(m.x.lb, 1)
    self.assertAlmostEqual(m.x.ub, 2)