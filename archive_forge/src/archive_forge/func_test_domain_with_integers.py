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
@parameterized.expand(input=_load_tests(mip_solvers))
def test_domain_with_integers(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-1, None), domain=pe.NonNegativeIntegers)
    m.obj = pe.Objective(expr=m.x)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 0)
    m.x.setlb(0.5)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 1)
    m.x.setlb(-5.5)
    m.x.domain = pe.Integers
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, -5)
    m.x.domain = pe.Binary
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 0)
    m.x.setlb(0.5)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 1)