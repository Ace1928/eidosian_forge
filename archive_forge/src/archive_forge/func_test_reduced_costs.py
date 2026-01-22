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
@parameterized.expand(input=_load_tests(all_solvers))
def test_reduced_costs(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-1, 1))
    m.y = pe.Var(bounds=(-2, 2))
    m.obj = pe.Objective(expr=3 * m.x + 4 * m.y)
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(m.x.value, -1)
    self.assertAlmostEqual(m.y.value, -2)
    rc = res.solution_loader.get_reduced_costs()
    self.assertAlmostEqual(rc[m.x], 3)
    self.assertAlmostEqual(rc[m.y], 4)
    m.obj.expr *= -1
    res = opt.solve(m)
    rc = res.solution_loader.get_reduced_costs()
    self.assertAlmostEqual(rc[m.x], -3)
    self.assertAlmostEqual(rc[m.y], -4)