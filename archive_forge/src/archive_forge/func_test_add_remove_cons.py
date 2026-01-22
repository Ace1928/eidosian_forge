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
def test_add_remove_cons(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
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
    a1 = -1
    a2 = 1
    b1 = 1
    b2 = 2
    a3 = 1
    b3 = 3
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.Constraint(expr=m.y >= a1 * m.x + b1)
    m.c2 = pe.Constraint(expr=m.y >= a2 * m.x + b2)
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
    self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
    self.assertAlmostEqual(res.incumbent_objective, m.y.value)
    if res.objective_bound is None:
        bound = -math.inf
    else:
        bound = res.objective_bound
    self.assertTrue(bound <= m.y.value)
    duals = res.solution_loader.get_duals()
    self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a2 - a1)))
    self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))
    m.c3 = pe.Constraint(expr=m.y >= a3 * m.x + b3)
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(m.x.value, (b3 - b1) / (a1 - a3))
    self.assertAlmostEqual(m.y.value, a1 * (b3 - b1) / (a1 - a3) + b1)
    self.assertAlmostEqual(res.incumbent_objective, m.y.value)
    self.assertTrue(res.objective_bound is None or res.objective_bound <= m.y.value)
    duals = res.solution_loader.get_duals()
    self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a3 - a1)))
    self.assertAlmostEqual(duals[m.c2], 0)
    self.assertAlmostEqual(duals[m.c3], a1 / (a3 - a1))
    del m.c3
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
    self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
    self.assertAlmostEqual(res.incumbent_objective, m.y.value)
    self.assertTrue(res.objective_bound is None or res.objective_bound <= m.y.value)
    duals = res.solution_loader.get_duals()
    self.assertAlmostEqual(duals[m.c1], -(1 + a1 / (a2 - a1)))
    self.assertAlmostEqual(duals[m.c2], a1 / (a2 - a1))