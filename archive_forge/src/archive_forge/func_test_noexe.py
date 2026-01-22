import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
def test_noexe(self):
    with SystemCallSolver(type='test') as opt:
        self.assertEqual(id(opt._user_executable), id(None))
        with self.assertRaises(NotImplementedError):
            opt.executable()