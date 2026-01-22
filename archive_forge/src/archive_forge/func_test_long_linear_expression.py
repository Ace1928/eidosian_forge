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
def test_long_linear_expression(self):
    N = 30
    for n in range(N):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=list(range(N)))
        m.x = pyo.Var(m.a, bounds=(0, 1))
        m.x[n].setub(None)
        m.c = pyo.Constraint(expr=LinearExpression(constant=0, linear_coefs=[1] * N, linear_vars=list(m.x.values())) == 1)
        self.tightener(m)
        self.assertAlmostEqual(m.x[n].ub, 1)
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=list(range(N)))
        m.x = pyo.Var(m.a, bounds=(0, 1))
        m.x[n].setlb(None)
        m.c = pyo.Constraint(expr=LinearExpression(constant=0, linear_coefs=[1] * N, linear_vars=list(m.x.values())) == 1)
        self.tightener(m)
        self.assertAlmostEqual(m.x[n].lb, -28)