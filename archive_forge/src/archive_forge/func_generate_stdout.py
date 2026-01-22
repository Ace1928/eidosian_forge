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
def generate_stdout(self, solver, version):
    if solver == 'scip':
        stdout = 'SCIP version {} [precision: 8 byte] [memory: block] [mode: optimized] [LP solver: SoPlex 6.0.0] [GitHash: d9b84b0709]\nCopyright (C) 2002-2021 Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)\n\nExternal libraries:\n   SoPlex 6.0.0    Linear Programming Solver developed at Zuse Institute Berlin (soplex.zib.de) [GitHash: f5cfa86b]'
    elif solver == 'scipampl':
        stdout = 'SCIP version {} [precision: 8 byte] [memory: block] [mode: optimized] [LP solver: SoPlex 5.0.2] [GitHash: 74c11e60cd]\nCopyright (C) 2002-2021 Konrad-Zuse-Zentrum fuer Informationstechnik Berlin (ZIB)\n\nExternal libraries:\n Readline 8.0         GNU library for command line editing (gnu.org/s/readline)'
    else:
        raise ValueError('Unsupported solver for stdout generation.')
    version = '.'.join((str(e) for e in version[:3]))
    return stdout.format(version)