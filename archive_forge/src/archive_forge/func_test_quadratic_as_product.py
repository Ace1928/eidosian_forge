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
def test_quadratic_as_product(self):
    m1 = pyo.ConcreteModel()
    m1.x = pyo.Var([1, 2], bounds=(-2, 6))
    m1.y = pyo.Var()
    m1.c = pyo.Constraint(expr=m1.x[1] * m1.x[1] + m1.x[2] * m1.x[2] == m1.y)
    m2 = pyo.ConcreteModel()
    m2.x = pyo.Var([1, 2], bounds=(-2, 6))
    m2.y = pyo.Var()
    m2.c = pyo.Constraint(expr=m2.x[1] ** 2 + m2.x[2] ** 2 == m2.y)
    self.tightener(m1)
    self.tightener(m2)
    self.assertAlmostEqual(m1.y.lb, m2.y.lb)
    self.assertAlmostEqual(m1.y.ub, m2.y.ub)
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2], bounds=(-2, 6))
    m.y = pyo.Var()
    m.c = pyo.Constraint(expr=m.x[1] * m.x[1] + m.x[2] * m.x[2] == 0)
    self.tightener(m)
    self.assertAlmostEqual(m.x[1].lb, 0)
    self.assertAlmostEqual(m.x[1].ub, 0)
    self.assertAlmostEqual(m.x[2].lb, 0)
    self.assertAlmostEqual(m.x[2].ub, 0)