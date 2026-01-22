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
def test_presolve_named_expressions(self):
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=1, bounds=(0, 10))
    m.subexpr = pyo.Expression(pyo.Integers)
    m.subexpr[1] = m.x[1] + m.x[2]
    m.eq = pyo.Constraint(pyo.Integers)
    m.eq[1] = m.x[1] == 7
    m.eq[2] = m.x[3] == 0.1 * m.subexpr[1] * m.x[2]
    m.obj = pyo.Objective(expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 3)
    OUT = io.StringIO()
    with LoggingIntercept() as LOG:
        nlinfo = nl_writer.NLWriter().write(m, OUT, symbolic_solver_labels=True, linear_presolve=True)
    self.assertEqual(LOG.getvalue(), '')
    self.assertEqual(nlinfo.eliminated_vars, [(m.x[1], 7)])
    self.assertEqual(*nl_diff("g3 1 1 0\t# problem unknown\n 2 1 1 0 1 \t# vars, constraints, objectives, ranges, eqns\n 1 1 0 0 0 0\t# nonlinear constrs, objs; ccons: lin, nonlin, nd, nzlb\n 0 0\t# network constraints: nonlinear, linear\n 1 2 1 \t# nonlinear vars in constraints, objectives, both\n 0 0 0 1\t# linear network variables; functions; arith, flags\n 0 0 0 0 0 \t# discrete variables: binary, integer, nonlinear (b,c,o)\n 2 2 \t# nonzeros in Jacobian, obj. gradient\n 5 4\t# max name lengths: constraints, variables\n 0 0 0 1 0\t# common exprs: b,c,o,c1,o1\nV2 1 1\t#subexpr[1]\n0 1\nn7.0\nC0\t#eq[2]\no16\t#-\no2\t#*\no2\t#*\nn0.1\nv2\t#subexpr[1]\nv0\t#x[2]\nO0 0\t#obj\no54\t# sumlist\n3\t# (n)\no5\t#^\nn7.0\nn2\no5\t#^\nv0\t#x[2]\nn2\no5\t#^\nv1\t#x[3]\nn3\nx2\t# initial guess\n0 1\t#x[2]\n1 1\t#x[3]\nr\t#1 ranges (rhs's)\n4 0\t#eq[2]\nb\t#2 bounds (on variables)\n0 0 10\t#x[2]\n0 0 10\t#x[3]\nk1\t#intermediate Jacobian column lengths\n1\nJ0 2\t#eq[2]\n0 0\n1 1\nG0 2\t#obj\n0 0\n1 0\n", OUT.getvalue()))