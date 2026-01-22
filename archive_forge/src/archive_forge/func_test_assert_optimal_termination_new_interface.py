from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.util import (
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from typing import Callable
from pyomo.common.gsl import find_GSL
from pyomo.opt.results import SolverResults
def test_assert_optimal_termination_new_interface(self):
    results = Results()
    results.solution_status = SolutionStatus.optimal
    results.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
    assert_optimal_termination(results)
    results.termination_condition = TerminationCondition.iterationLimit
    with self.assertRaises(RuntimeError):
        assert_optimal_termination(results)
    results.solution_status = SolutionStatus.noSolution
    with self.assertRaises(RuntimeError):
        assert_optimal_termination(results)