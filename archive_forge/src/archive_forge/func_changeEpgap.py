from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def changeEpgap(self, epgap=10 ** (-4)):
    """
            Change cplex solver integer bound gap tolerence
            """
    self.solverModel.parameters.mip.tolerances.mipgap.set(epgap)