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
def setParam(self, name, val):
    """
            Set parameter to COPT problem
            """
    par_type = ctypes.c_int()
    par_name = coptstr(name)
    rc = self.SearchParamAttr(self.coptprob, par_name, byref(par_type))
    if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(par_name))
    if par_type.value == 0:
        rc = self.SetDblParam(self.coptprob, par_name, ctypes.c_double(val))
        if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to set double parameter '{}'".format(par_name))
    elif par_type.value == 1:
        rc = self.SetIntParam(self.coptprob, par_name, ctypes.c_int(val))
        if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to set integer parameter '{}'".format(par_name))
    else:
        raise PulpSolverError("COPT_PULP: Invalid parameter '{}'".format(par_name))