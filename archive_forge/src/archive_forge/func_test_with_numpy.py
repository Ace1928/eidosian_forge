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
def test_with_numpy(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
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
    m.obj = pe.Objective(expr=m.y)
    a1 = 1
    b1 = 3
    a2 = -2
    b2 = 1
    m.c1 = pe.Constraint(expr=(np.float64(0), m.y - np.int64(1) * m.x - np.float32(3), None))
    m.c2 = pe.Constraint(expr=(None, -m.y + np.int32(-2) * m.x + np.float64(1), np.float16(0)))
    res = opt.solve(m)
    self.assertEqual(res.solution_status, SolutionStatus.optimal)
    self.assertAlmostEqual(m.x.value, (b2 - b1) / (a1 - a2))
    self.assertAlmostEqual(m.y.value, a1 * (b2 - b1) / (a1 - a2) + b1)