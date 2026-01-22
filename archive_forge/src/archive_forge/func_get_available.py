import subprocess
import sys
from os.path import join, exists, splitext
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ
from pyomo.opt import SolverFactory
from pyomo.core import ConcreteModel, Var, Objective, Constraint
import pyomo.solvers.plugins.solvers.SCIPAMPL
def get_available(*args, **kwargs):
    name = args[0]._registered_name
    if name in executables:
        return executables[name] is not None
    elif fail:
        self.fail('Solver creation looked up a non scip executable.')
    else:
        return False