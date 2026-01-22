import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def test_write_without_priority_suffix(self):
    with self.assertRaises(ValueError):
        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)