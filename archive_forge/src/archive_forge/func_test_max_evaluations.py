import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
def test_max_evaluations(self):
    """
        solver_kwargs={'options': {'maxNodes': 1}}
        """
    lp_file = 'max_evaluations.out.lp'
    results = self.opt.solve(os.path.join(data_dir, lp_file))
    self.assertEqual(1.2052223, results.problem.lower_bound)
    self.assertEqual(1.20645976, results.problem.upper_bound)
    self.assertEqual(SolverStatus.aborted, results.solver.status)
    self.assertEqual(0.16, results.solver.system_time)
    self.assertEqual(0.18, results.solver.wallclock_time)
    self.assertEqual(TerminationCondition.maxEvaluations, results.solver.termination_condition)
    self.assertEqual('Optimization terminated because the total number of branch-and-cut nodes explored exceeded the value specified in the maxNodes parameter', results.solver.termination_message)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 1)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 1)
    self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 602)