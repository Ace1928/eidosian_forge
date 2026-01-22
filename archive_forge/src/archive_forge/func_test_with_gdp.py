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
def test_with_gdp(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
    opt: SolverBase = opt_class()
    if not opt.available():
        raise unittest.SkipTest(f'Solver {opt.name} not available.')
    if any((name.startswith(i) for i in nl_solvers_set)):
        if use_presolve:
            opt.config.writer_config.linear_presolve = True
        else:
            opt.config.writer_config.linear_presolve = False
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-10, 10))
    m.y = pe.Var(bounds=(-10, 10))
    m.obj = pe.Objective(expr=m.y)
    m.d1 = gdp.Disjunct()
    m.d1.c1 = pe.Constraint(expr=m.y >= m.x + 2)
    m.d1.c2 = pe.Constraint(expr=m.y >= -m.x + 2)
    m.d2 = gdp.Disjunct()
    m.d2.c1 = pe.Constraint(expr=m.y >= m.x + 1)
    m.d2.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
    m.disjunction = gdp.Disjunction(expr=[m.d2, m.d1])
    pe.TransformationFactory('gdp.bigm').apply_to(m)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 1)
    self.assertAlmostEqual(m.x.value, 0)
    self.assertAlmostEqual(m.y.value, 1)
    opt: SolverBase = opt_class()
    opt.use_extensions = True
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 1)
    self.assertAlmostEqual(m.x.value, 0)
    self.assertAlmostEqual(m.y.value, 1)