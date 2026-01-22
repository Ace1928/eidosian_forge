import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
def test_executable_isexe_abspath(self):
    for name in _test_names:
        with SolverFactory(name, executable=isexe_abspath) as opt:
            if isinstance(opt, UnknownSolver):
                continue
            self.assertEqual(opt._user_executable, isexe_abspath)
            self.assertEqual(opt.executable(), isexe_abspath)