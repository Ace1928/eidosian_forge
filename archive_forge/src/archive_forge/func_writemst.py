import os
import sys
import ctypes
import subprocess
import warnings
from uuid import uuid4
from .core import sparse, ctypesArrayFill, PulpSolverError
from .core import clock, log
from .core import LpSolver, LpSolver_CMD
from ..constants import (
from ..constants import LpContinuous, LpBinary, LpInteger
from ..constants import LpConstraintEQ, LpConstraintLE, LpConstraintGE
from ..constants import LpMinimize, LpMaximize
def writemst(self, filename, lpvars):
    """
        Write COPT MIP start file
        """
    mstvals = [(v.name, v.value()) for v in lpvars if v.value() is not None]
    mstline = []
    for varname, varval in mstvals:
        mstline.append('{0} {1}'.format(varname, varval))
    with open(filename, 'w') as mstfile:
        mstfile.write('\n'.join(mstline))
    return True