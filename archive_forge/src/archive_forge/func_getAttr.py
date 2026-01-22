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
def getAttr(self, name):
    """
            Get attribute of the problem
            """
    attr_dblval = ctypes.c_double()
    attr_intval = ctypes.c_int()
    attr_type = ctypes.c_int()
    attr_name = coptstr(name)
    rc = self.SearchParamAttr(self.coptprob, attr_name, byref(attr_type))
    if rc != 0:
        raise PulpSolverError("COPT_PULP: Failed to check type for '{}'".format(attr_name))
    if attr_type.value == 2:
        rc = self.GetDblAttr(self.coptprob, attr_name, byref(attr_dblval))
        if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to get double attribute '{}'".format(attr_name))
        else:
            retval = attr_dblval.value
    elif attr_type.value == 3:
        rc = self.GetIntAttr(self.coptprob, attr_name, byref(attr_intval))
        if rc != 0:
            raise PulpSolverError("COPT_PULP: Failed to get integer attribute '{}'".format(attr_name))
        else:
            retval = attr_intval.value
    else:
        raise PulpSolverError("COPT_PULP: Invalid attribute '{}'".format(attr_name))
    return retval