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
def test_variables_elsewhere(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
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
    m.b = pe.Block()
    m.b.obj = pe.Objective(expr=m.y)
    m.b.c1 = pe.Constraint(expr=m.y >= m.x + 2)
    m.b.c2 = pe.Constraint(expr=m.y >= -m.x)
    res = opt.solve(m.b)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(res.incumbent_objective, 1)
    self.assertAlmostEqual(m.x.value, -1)
    self.assertAlmostEqual(m.y.value, 1)
    m.x.setlb(0)
    res = opt.solve(m.b)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(res.incumbent_objective, 2)
    self.assertAlmostEqual(m.x.value, 0)
    self.assertAlmostEqual(m.y.value, 2)