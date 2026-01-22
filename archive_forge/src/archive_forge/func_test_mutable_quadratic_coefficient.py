import random
import math
from typing import Type
import pyomo.environ as pe
from pyomo import gdp
from pyomo.common.dependencies import attempt_import
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.results import TerminationCondition, SolutionStatus, Results
from pyomo.contrib.solver.base import SolverBase
from pyomo.contrib.solver.ipopt import Ipopt
from pyomo.contrib.solver.gurobi import Gurobi
from pyomo.core.expr.numeric_expr import LinearExpression
@parameterized.expand(input=_load_tests(qcp_solvers))
def test_mutable_quadratic_coefficient(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var()
    m.y = pe.Var()
    m.a = pe.Param(initialize=1, mutable=True)
    m.b = pe.Param(initialize=-1, mutable=True)
    m.obj = pe.Objective(expr=m.x ** 2 + m.y ** 2)
    m.c = pe.Constraint(expr=m.y >= (m.a * m.x + m.b) ** 2)
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 0.41024548525899274, 4)
    self.assertAlmostEqual(m.y.value, 0.34781038127030117, 4)
    m.a.value = 2
    m.b.value = -0.5
    res = opt.solve(m)
    self.assertAlmostEqual(m.x.value, 0.10256137418973625, 4)
    self.assertAlmostEqual(m.y.value, 0.0869525991355825, 4)