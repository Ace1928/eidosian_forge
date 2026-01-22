from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .core import cbc_path, pulp_cbc_path, coinMP_path, devnull, operating_system
import os
from .. import constants
from tempfile import mktemp
import ctypes
import warnings
def readsol_LP(self, filename, lp, vs):
    """
        Read a CBC solution file generated from an lp (good names)
        returns status, values, reducedCosts, shadowPrices, slacks, sol_status
        """
    variablesNames = {v.name: v.name for v in vs}
    constraintsNames = {c: c for c in lp.constraints}
    return self.readsol_MPS(filename, lp, vs, variablesNames, constraintsNames)