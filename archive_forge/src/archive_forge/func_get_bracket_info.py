import math
import warnings
import sys
import inspect
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
import numpy as np
from scipy.linalg import cholesky, issymmetric, LinAlgError
from scipy.sparse.linalg import LinearOperator
from ._linesearch import (line_search_wolfe1, line_search_wolfe2,
from ._numdiff import approx_derivative
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy._lib._util import MapWrapper, check_random_state
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
def get_bracket_info(self):
    func = self.func
    args = self.args
    brack = self.brack
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, xa=brack[0], xb=brack[1], args=args)
    elif len(brack) == 3:
        xa, xb, xc = brack
        if xa > xc:
            xc, xa = (xa, xc)
        if not (xa < xb and xb < xc):
            raise ValueError('Bracketing values (xa, xb, xc) do not fulfill this requirement: (xa < xb) and (xb < xc)')
        fa = func(*(xa,) + args)
        fb = func(*(xb,) + args)
        fc = func(*(xc,) + args)
        if not (fb < fa and fb < fc):
            raise ValueError('Bracketing values (xa, xb, xc) do not fulfill this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))')
        funcalls = 3
    else:
        raise ValueError('Bracketing interval must be length 2 or 3 sequence.')
    return (xa, xb, xc, fa, fb, fc, funcalls)