from pyomo.common import unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.util import (
from pyomo.contrib.solver.results import Results, SolutionStatus, TerminationCondition
from typing import Callable
from pyomo.common.gsl import find_GSL
from pyomo.opt.results import SolverResults
def test_assert_optimal_termination_legacy_interface(self):
    results = SolverResults()
    results.solver.status = SolverStatus.ok
    results.solver.termination_condition = LegacyTerminationCondition.optimal
    assert_optimal_termination(results)
    results.solver.termination_condition = LegacyTerminationCondition.unknown
    with self.assertRaises(RuntimeError):
        assert_optimal_termination(results)
    results.solver.termination_condition = SolverStatus.aborted
    with self.assertRaises(RuntimeError):
        assert_optimal_termination(results)