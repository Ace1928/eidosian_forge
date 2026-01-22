import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
def test_fix_parsing_bug(self):
    """
        The test wasn't generated using the method in the class docstring
        See https://github.com/Pyomo/pyomo/issues/1001
        """
    lp_file = 'fix_parsing_bug.out.lp'
    results = self.opt.solve(os.path.join(data_dir, lp_file))
    if self.opt.version() < (2, 10, 2):
        self.assertEqual(3.0, results.problem.lower_bound)
        self.assertEqual(3.0, results.problem.upper_bound)
    else:
        self.assertEqual(-3.0, results.problem.lower_bound)
        self.assertEqual(-3.0, results.problem.upper_bound)
    self.assertEqual(SolverStatus.aborted, results.solver.status)
    self.assertEqual(0.08, results.solver.system_time)
    self.assertEqual(0.09, results.solver.wallclock_time)
    self.assertEqual(TerminationCondition.other, results.solver.termination_condition)
    self.assertEqual('Optimization terminated because the number of solutions found reached the value specified in the maxSolutions parameter.', results.solver.termination_message)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
    self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 0)