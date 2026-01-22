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
def test_objective_changes(self, name: str, opt_class: Type[SolverBase], use_presolve: bool):
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
    m.c1 = pe.Constraint(expr=m.y >= m.x + 1)
    m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
    m.obj = pe.Objective(expr=m.y)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 1)
    del m.obj
    m.obj = pe.Objective(expr=2 * m.y)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 2)
    m.obj.expr = 3 * m.y
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 3)
    m.obj.sense = pe.maximize
    opt.config.raise_exception_on_nonoptimal_result = False
    opt.config.load_solutions = False
    res = opt.solve(m)
    self.assertIn(res.termination_condition, {TerminationCondition.unbounded, TerminationCondition.infeasibleOrUnbounded})
    m.obj.sense = pe.minimize
    opt.config.load_solutions = True
    del m.obj
    m.obj = pe.Objective(expr=m.x * m.y)
    m.x.fix(2)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 6, 6)
    m.x.fix(3)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 12, 6)
    m.x.unfix()
    m.y.fix(2)
    m.x.setlb(-3)
    m.x.setub(5)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, -2, 6)
    m.y.unfix()
    m.x.setlb(None)
    m.x.setub(None)
    m.e = pe.Expression(expr=2)
    del m.obj
    m.obj = pe.Objective(expr=m.e * m.y)
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 2)
    m.e.expr = 3
    res = opt.solve(m)
    self.assertAlmostEqual(res.incumbent_objective, 3)
    if opt.is_persistent():
        opt.config.auto_updates.check_for_new_objective = False
        m.e.expr = 4
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, 4)