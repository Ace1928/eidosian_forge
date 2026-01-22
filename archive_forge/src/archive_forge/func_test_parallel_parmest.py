from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from itertools import product
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverFactory
from pyomo.common.fileutils import find_library
@unittest.skip('Presently having trouble with mpiexec on appveyor')
def test_parallel_parmest(self):
    """use mpiexec and mpi4py"""
    p = str(parmestbase.__path__)
    l = p.find("'")
    r = p.find("'", l + 1)
    parmestpath = p[l + 1:r]
    rbpath = parmestpath + os.sep + 'examples' + os.sep + 'rooney_biegler' + os.sep + 'rooney_biegler_parmest.py'
    rbpath = os.path.abspath(rbpath)
    rlist = ['mpiexec', '--allow-run-as-root', '-n', '2', sys.executable, rbpath]
    if sys.version_info >= (3, 5):
        ret = subprocess.run(rlist)
        retcode = ret.returncode
    else:
        retcode = subprocess.call(rlist)
    assert retcode == 0