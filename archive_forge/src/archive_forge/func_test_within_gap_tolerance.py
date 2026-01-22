import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
def test_within_gap_tolerance(self):
    """
        solver_kwargs={'options': {'allowableGap': 1000000}}
        """
    lp_file = 'within_gap_tolerance.out.lp'
    results = self.opt.solve(os.path.join(data_dir, lp_file))
    self.assertEqual(0.925437, results.problem.lower_bound)
    self.assertEqual(1.35481947, results.problem.upper_bound)
    self.assertEqual(SolverStatus.ok, results.solver.status)
    self.assertEqual(0.07, results.solver.system_time)
    self.assertEqual(0.07, results.solver.wallclock_time)
    self.assertEqual(TerminationCondition.optimal, results.solver.termination_condition)
    self.assertEqual('Model was solved to optimality (subject to tolerances), and an optimal solution is available.', results.solver.termination_message)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
    self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 0)