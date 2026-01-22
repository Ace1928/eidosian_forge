from .core import LpSolver, LpSolver_CMD, subprocess, PulpSolverError
from .. import constants
import warnings
import sys
import re
@staticmethod
def quote_path(path):
    """
        Quotes a path for the Xpress optimizer console, by wrapping it in
        double quotes and escaping the following characters, which would
        otherwise be interpreted by the Tcl shell: \\ $ " [
        """
    return '"' + re.sub('([\\\\$"[])', '\\\\\\1', path) + '"'