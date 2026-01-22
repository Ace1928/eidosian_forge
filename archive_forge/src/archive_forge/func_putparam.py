from .core import LpSolver, PulpSolverError
from .. import constants
import sys
from typing import Optional
def putparam(self, par, val):
    """
            Pass the values of valid parameters to Mosek.
            """
    if isinstance(par, mosek.dparam):
        self.task.putdouparam(par, val)
    elif isinstance(par, mosek.iparam):
        self.task.putintparam(par, val)
    elif isinstance(par, mosek.sparam):
        self.task.putstrparam(par, val)
    elif isinstance(par, str):
        if par.startswith('MSK_DPAR_'):
            self.task.putnadouparam(par, val)
        elif par.startswith('MSK_IPAR_'):
            self.task.putnaintparam(par, val)
        elif par.startswith('MSK_SPAR_'):
            self.task.putnastrparam(par, val)
        else:
            raise PulpSolverError("Invalid MOSEK parameter: '{}'. Check MOSEK documentation for a list of valid parameters.".format(par))