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
def test_stale_vars(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
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
    m.z = pe.Var()
    m.obj = pe.Objective(expr=m.y)
    m.c1 = pe.Constraint(expr=m.y >= m.x)
    m.c2 = pe.Constraint(expr=m.y >= -m.x)
    m.x.value = 1
    m.y.value = 1
    m.z.value = 1
    self.assertFalse(m.x.stale)
    self.assertFalse(m.y.stale)
    self.assertFalse(m.z.stale)
    res = opt.solve(m)
    self.assertFalse(m.x.stale)
    self.assertFalse(m.y.stale)
    self.assertTrue(m.z.stale)
    opt.config.load_solutions = False
    res = opt.solve(m)
    self.assertTrue(m.x.stale)
    self.assertTrue(m.y.stale)
    self.assertTrue(m.z.stale)
    res.solution_loader.load_vars()
    self.assertFalse(m.x.stale)
    self.assertFalse(m.y.stale)
    self.assertTrue(m.z.stale)
    res = opt.solve(m)
    self.assertTrue(m.x.stale)
    self.assertTrue(m.y.stale)
    self.assertTrue(m.z.stale)
    res.solution_loader.load_vars([m.y])
    self.assertFalse(m.y.stale)