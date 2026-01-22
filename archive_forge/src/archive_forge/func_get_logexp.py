from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def get_logexp(a=1, b=0, a2=None, b2=None, backend=None):
    """ Utility function for use with :func:symmetricsys.

    Creates a pair of callbacks for logarithmic transformation
    (including scaling and shifting): ``u = ln(a*x + b)``.

    Parameters
    ----------
    a : number
        Scaling (forward).
    b : number
        Shift (forward).
    a2 : number
        Scaling (backward).
    b2 : number
        Shift (backward).

    Returns
    -------
    Pair of callbacks.

    """
    if a2 is None:
        a2 = a
    if b2 is None:
        b2 = b
    if backend is None:
        import sympy as backend
    return (lambda x: backend.log(a * x + b), lambda x: (backend.exp(x) - b2) / a2)