import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.environ import (
from pyomo.opt import WriterFactory, SolverFactory
def test_solve_infeasible_problem(self):
    m = ConcreteModel()
    m.x = Var(within=[1, 2, 3, 5])
    m.c = Constraint(expr=m.x == 0)
    result = SolverFactory('cp_optimizer').solve(m)
    self.assertEqual(result.solver.termination_condition, TerminationCondition.infeasible)
    self.assertIsNone(m.x.value)