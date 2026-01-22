import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def test_use_variable_priorities(self):
    model = self.get_mock_model_with_priorities()
    with SolverFactory('_mock_cplex') as opt:
        opt._presolve(model, priorities=True, keepfiles=True, symbolic_solver_labels=True)
        with open(opt._priorities_file_name, 'r') as ord_file:
            priorities_file = ord_file.read()
    self.assertEqual(priorities_file, '* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x 1\n DN y(0) 2\n DN y(1) 2\n DN y(2) 2\n DN y(3) 2\n DN y(4) 2\n DN y(5) 2\n DN y(6) 2\n DN y(7) 2\n DN y(8) 2\n UP y(9) 2\nENDATA\n')
    self.assertIn('read %s\n' % (opt._priorities_file_name,), opt._command.script)