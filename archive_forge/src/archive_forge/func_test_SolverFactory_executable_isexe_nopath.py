import os
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.opt.base import UnknownSolver
from pyomo.opt.base.solvers import SolverFactory
from pyomo.opt.solver import SystemCallSolver
def test_SolverFactory_executable_isexe_nopath(self):
    for name in _test_names:
        with SolverFactory(name, executable=isexe_nopath) as opt:
            self.assertTrue(isinstance(opt, UnknownSolver))
    rm_PATH = False
    orig_PATH = None
    if 'PATH' in os.environ:
        orig_PATH = os.environ['PATH']
    else:
        rm_PATH = True
        os.environ['PATH'] = ''
    os.environ['PATH'] = exedir + os.pathsep + os.environ['PATH']
    try:
        for name in _test_names:
            with SolverFactory(name, executable=isexe_nopath) as opt:
                if isinstance(opt, UnknownSolver):
                    continue
                self.assertEqual(opt._user_executable, isexe_abspath)
                self.assertEqual(opt.executable(), isexe_abspath)
    finally:
        if rm_PATH:
            del os.environ['PATH']
        else:
            os.environ['PATH'] = orig_PATH