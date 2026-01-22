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
def test_add_and_remove_vars(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.y = pe.Var(bounds=(-1, None))
    m.obj = pe.Objective(expr=m.y)
    if opt.is_persistent():
        opt.config.auto_updates.update_parameters = False
        opt.config.auto_updates.update_vars = False
        opt.config.auto_updates.update_constraints = False
        opt.config.auto_updates.update_named_expressions = False
        opt.config.auto_updates.check_for_new_or_removed_params = False
        opt.config.auto_updates.check_for_new_or_removed_constraints = False
        opt.config.auto_updates.check_for_new_or_removed_vars = False
    opt.config.load_solutions = False
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    res.solution_loader.load_vars()
    self.assertAlmostEqual(m.y.value, -1)
    m.x = pe.Var()
    a1 = 1
    a2 = -1
    b1 = 2
    b2 = 1
    m.c1 = pe.Constraint(expr=(0, m.y - a1 * m.x - b1, None))
    m.c2 = pe.Constraint(expr=(None, -m.y + a2 * m.x + b2, 0))
    if opt.is_persistent():
        opt.add_constraints([m.c1, m.c2])
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    res.solution_loader.load_vars()
    self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
    self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)
    m.c1.deactivate()
    m.c2.deactivate()
    if opt.is_persistent():
        opt.remove_constraints([m.c1, m.c2])
    m.x.value = None
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    res.solution_loader.load_vars()
    self.assertEqual(m.x.value, None)
    self.assertAlmostEqual(m.y.value, -1)