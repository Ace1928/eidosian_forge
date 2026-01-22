from io import StringIO
import shlex
from tempfile import mkdtemp
import os, sys, math, logging, shutil, time, subprocess
from pyomo.core.base import Constraint, Var, value, Objective
from pyomo.opt import ProblemFormat, SolverFactory
import pyomo.common
from pyomo.common.collections import Bunch
from pyomo.common.tee import TeeStream
from pyomo.opt.base.solvers import _extract_version
from pyomo.core.kernel.block import IBlock
from pyomo.core.kernel.objective import IObjective
from pyomo.core.kernel.variable import IVariable
import pyomo.core.base.suffix
import pyomo.core.kernel.suffix
from pyomo.opt.results import (
from pyomo.common.dependencies import attempt_import
def file_removal_gams_direct(tmpdir, newdir):
    if newdir:
        shutil.rmtree(tmpdir)
    else:
        os.remove(os.path.join(tmpdir, '_gams_py_gjo0.gms'))
        os.remove(os.path.join(tmpdir, '_gams_py_gjo0.lst'))
        os.remove(os.path.join(tmpdir, '_gams_py_gdb0.gdx'))