from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def setThreads(self, threads=None):
    """
            Change cplex thread count used (None is default which uses all available resources)
            """
    self.solverModel.parameters.threads.set(threads or 0)