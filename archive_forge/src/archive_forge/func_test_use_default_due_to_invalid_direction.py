import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def test_use_default_due_to_invalid_direction(self):
    self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
    priority_val = 10
    self._set_suffix_value(self.mock_model.priority, self.mock_model.x, priority_val)
    self.mock_model.direction = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
    self._set_suffix_value(self.mock_model.direction, self.mock_model.x, 'invalid_branching_direction')
    CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
    priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)
    self.assertEqual(priorities_file, '* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x 10\nENDATA\n')