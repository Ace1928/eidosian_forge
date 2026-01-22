import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_dont_skip_trivial_and_call_count_for_unfixed_con_is_one(self):
    self.setup(skip_trivial_constraints=False)
    self.assertFalse(self._opt._skip_trivial_constraints)
    self.assertFalse(self._model.c2.body.is_fixed())
    with unittest.mock.patch('pyomo.solvers.plugins.solvers.cplex_direct.is_fixed', wraps=is_fixed) as mock_is_fixed:
        self.assertEqual(mock_is_fixed.call_count, 0)
        self._opt.add_constraint(self._model.c2)
        self.assertEqual(mock_is_fixed.call_count, 1)