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
def test_always_feasible(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(1, 2))
    m.y = pyo.Var(bounds=(1, 2))
    m.c = pyo.Constraint(expr=m.x + m.y >= 0)
    self.tightener(m)
    self.assertTrue(m.c.active)
    if self.tightener is fbbt:
        self.tightener(m, deactivate_satisfied_constraints=True)
    else:
        self.it.config.deactivate_satisfied_constraints = True
        self.tightener(m)
        self.it.config.deactivate_satisfied_constraints = False
    self.assertFalse(m.c.active)