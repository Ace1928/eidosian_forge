from .core import LpSolver, PulpSolverError
from .. import constants
import sys
from typing import Optional
def setOutStream(self, text):
    """Sets the log-output stream."""
    sys.stdout.write(text)
    sys.stdout.flush()