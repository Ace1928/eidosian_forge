import functools
import re
from .ctx_base import StandardBaseContext
from .libmp.backend import basestring, BACKEND
from . import libmp
from .libmp import (MPZ, MPZ_ZERO, MPZ_ONE, int_types, repr_dps,
from . import function_docs
from . import rational
from .ctx_mp_python import _mpf, _mpc, mpnumeric
def workdps(ctx, n, normalize_output=False):
    """
        This function is analogous to workprec (see documentation)
        but changes the decimal precision instead of the number of bits.
        """
    return PrecisionManager(ctx, None, lambda d: n, normalize_output)