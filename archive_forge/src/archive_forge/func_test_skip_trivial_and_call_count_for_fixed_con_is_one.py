import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_skip_trivial_and_call_count_for_fixed_con_is_one(self):
    self.setup(skip_trivial_constraints=True)
    self._model.x.fix(1)
    self.assertTrue(self._opt._skip_trivial_constraints)
    self.assertTrue(self._model.c2.body.is_fixed())
    with unittest.mock.patch('pyomo.solvers.plugins.solvers.cplex_direct.is_fixed', wraps=is_fixed) as mock_is_fixed:
        self.assertEqual(mock_is_fixed.call_count, 0)
        self._opt.add_constraint(self._model.c2)
        self.assertEqual(mock_is_fixed.call_count, 1)