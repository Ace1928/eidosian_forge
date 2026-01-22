import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
def test_executable_isexe_abspath_user(self):
    for name in _test_names:
        with SolverFactory(name, executable=isexe_abspath_user) as opt:
            if isinstance(opt, UnknownSolver):
                continue
            self.assertTrue(os.path.samefile(opt._user_executable, isexe_abspath))
            self.assertTrue(os.path.samefile(opt.executable(), isexe_abspath))