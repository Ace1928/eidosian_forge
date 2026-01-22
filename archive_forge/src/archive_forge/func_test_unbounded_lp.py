import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
@unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
def test_unbounded_lp(self):
    self.model.Idx = RangeSet(2)
    self.model.X = Var(self.model.Idx, within=Reals)
    self.model.Obj = Objective(expr=self.model.X[1] + self.model.X[2], sense=maximize)
    results = self.opt.solve(self.model)
    self.assertEqual(ProblemSense.maximize, results.problem.sense)
    self.assertEqual(TerminationCondition.unbounded, results.solver.termination_condition)
    self.assertEqual('Model was proven to be unbounded.', results.solver.termination_message)
    self.assertEqual(SolverStatus.warning, results.solver.status)