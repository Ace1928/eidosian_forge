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
def test_encountered_bugs3(self):
    xl = 0.033689710575092756
    xu = 0.04008169994804723
    yl = 0.03369608678342047
    yu = 0.04009243987444148
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(xl, xu))
    m.y = pyo.Var(bounds=(yl, yu))
    m.c = pyo.Constraint(expr=m.x == pyo.sin(m.y))
    self.tightener(m)
    self.assertAlmostEqual(m.x.lb, xl)
    self.assertAlmostEqual(m.x.ub, xu)
    self.assertAlmostEqual(m.y.lb, yl)
    self.assertAlmostEqual(m.y.ub, yu)