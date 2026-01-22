import pyomo.common.unittest as unittest
import io
import logging
import math
import os
import re
import pyomo.repn.util as repn_util
import pyomo.repn.plugins.nl_writer as nl_writer
from pyomo.repn.util import InvalidNumber
from pyomo.repn.tests.nl_diff import nl_diff
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.errors import MouseTrap
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.timing import report_timing
from pyomo.core.expr import Expr_if, inequality, LinearExpression
from pyomo.core.base.expression import ScalarExpression
from pyomo.environ import (
import pyomo.environ as pyo
def test_presolve_lower_triangular_out_of_bounds(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(5), domain=pyo.NonNegativeReals)
    m.obj = Objective(expr=m.x[3] + m.x[4])
    m.c = pyo.ConstraintList()
    m.c.add(m.x[0] == 5)
    m.c.add(2 * m.x[0] + 3 * m.x[2] == 19)
    m.c.add(m.x[0] + 2 * m.x[2] - 2 * m.x[1] == 3)
    m.c.add(-2 * m.x[0] + m.x[2] + m.x[1] - m.x[3] == 1)
    OUT = io.StringIO()
    with self.assertRaisesRegex(nl_writer.InfeasibleConstraintException, "model contains a trivially infeasible variable 'x\\[3\\]' \\(presolved to a value of -4.0 outside bounds \\[0, None\\]\\)."):
        with LoggingIntercept() as LOG:
            nlinfo = nl_writer.NLWriter().write(m, OUT, linear_presolve=True)
    self.assertEqual(LOG.getvalue(), '')