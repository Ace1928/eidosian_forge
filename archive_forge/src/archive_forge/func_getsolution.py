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
def getsolution(self, lp, ncols, nrows):
    """Get problem solution

            This function borrowed implementation of CPLEX_DLL.findSolutionValues,
            with some modifications.
            """
    status = ctypes.c_int()
    x = (ctypes.c_double * ncols)()
    dj = (ctypes.c_double * ncols)()
    pi = (ctypes.c_double * nrows)()
    slack = (ctypes.c_double * nrows)()
    var_x = {}
    var_dj = {}
    con_pi = {}
    con_slack = {}
    if lp.isMIP() and self.mip:
        hasmipsol = ctypes.c_int()
        rc = self.GetIntAttr(self.coptprob, coptstr('MipStatus'), byref(status))
        if rc != 0:
            raise PulpSolverError('COPT_PULP: Failed to get MIP status')
        rc = self.GetIntAttr(self.coptprob, coptstr('HasMipSol'), byref(hasmipsol))
        if rc != 0:
            raise PulpSolverError('COPT_PULP: Failed to check if MIP solution exists')
        if status.value == 1 or hasmipsol.value == 1:
            rc = self.GetSolution(self.coptprob, byref(x))
            if rc != 0:
                raise PulpSolverError('COPT_PULP: Failed to get MIP solution')
            for i in range(ncols):
                var_x[self.n2v[i].name] = x[i]
        lp.assignVarsVals(var_x)
    else:
        rc = self.GetIntAttr(self.coptprob, coptstr('LpStatus'), byref(status))
        if rc != 0:
            raise PulpSolverError('COPT_PULP: Failed to get LP status')
        if status.value == 1:
            rc = self.GetLpSolution(self.coptprob, byref(x), byref(slack), byref(pi), byref(dj))
            if rc != 0:
                raise PulpSolverError('COPT_PULP: Failed to get LP solution')
            for i in range(ncols):
                var_x[self.n2v[i].name] = x[i]
                var_dj[self.n2v[i].name] = dj[i]
            for i in range(nrows):
                con_pi[self.n2c[i]] = pi[i]
                con_slack[self.n2c[i]] = slack[i]
        lp.assignVarsVals(var_x)
        lp.assignVarsDj(var_dj)
        lp.assignConsPi(con_pi)
        lp.assignConsSlack(con_slack)
    lp.resolveOK = True
    for var in lp.variables():
        var.isModified = False
    lp.status = coptlpstat.get(status.value, LpStatusUndefined)
    return lp.status