import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.environ import (
from pyomo.opt import WriterFactory, SolverFactory
def test_solve_max_problem(self):
    m = ConcreteModel()
    m.cookies = Var(domain=PositiveIntegers, bounds=(7, 10))
    m.chocolate_chip_equity = Constraint(expr=m.cookies <= 9)
    m.obj = Objective(expr=m.cookies, sense=maximize)
    results = SolverFactory('cp_optimizer').solve(m)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
    self.assertEqual(value(m.cookies), 9)
    self.assertEqual(results.problem.number_of_objectives, 1)
    self.assertEqual(results.problem.sense, maximize)
    self.assertEqual(results.problem.lower_bound, 9)
    self.assertEqual(results.problem.upper_bound, 9)