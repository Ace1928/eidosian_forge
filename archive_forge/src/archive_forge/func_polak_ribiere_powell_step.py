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
def polak_ribiere_powell_step(alpha, gfkp1=None):
    xkp1 = xk + alpha * pk
    if gfkp1 is None:
        gfkp1 = myfprime(xkp1)
    yk = gfkp1 - gfk
    beta_k = max(0, np.dot(yk, gfkp1) / deltak)
    pkp1 = -gfkp1 + beta_k * pk
    gnorm = vecnorm(gfkp1, ord=norm)
    return (alpha, xkp1, pkp1, gfkp1, gnorm)