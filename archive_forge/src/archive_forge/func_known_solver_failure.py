import sys
import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.tests.eight_process_problem import EightProcessFlowsheet
from pyomo.contrib.mindtpy.tests.MINLP_simple import SimpleMINLP as SimpleMINLP
from pyomo.contrib.mindtpy.tests.MINLP3_simple import SimpleMINLP as SimpleMINLP3
from pyomo.contrib.mindtpy.tests.constraint_qualification_example import (
from pyomo.environ import SolverFactory, value
from pyomo.opt import TerminationCondition
def known_solver_failure(mip_solver, model):
    if mip_solver == 'gurobi_persistent' and model.name in {'DuranEx3', 'SimpleMINLP'} and sys.platform.startswith('win') and (SolverFactory(mip_solver).version()[:3] == (9, 5, 0)):
        sys.stderr.write(f'Skipping sub-test {model.name} with {mip_solver} due to known failure when running Gurobi 9.5.0 on Windows\n')
        return True
    return False