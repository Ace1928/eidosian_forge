from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
def vectorize1(func, args=(), vec_func=False):
    """Vectorize the call to a function.

    This is an internal utility function used by `romberg` and
    `quadrature` to create a vectorized version of a function.

    If `vec_func` is True, the function `func` is assumed to take vector
    arguments.

    Parameters
    ----------
    func : callable
        User defined function.
    args : tuple, optional
        Extra arguments for the function.
    vec_func : bool, optional
        True if the function func takes vector arguments.

    Returns
    -------
    vfunc : callable
        A function that will take a vector argument and return the
        result.

    """
    if vec_func:

        def vfunc(x):
            return func(x, *args)
    else:

        def vfunc(x):
            if np.isscalar(x):
                return func(x, *args)
            x = np.asarray(x)
            y0 = func(x[0], *args)
            n = len(x)
            dtype = getattr(y0, 'dtype', type(y0))
            output = np.empty((n,), dtype=dtype)
            output[0] = y0
            for i in range(1, n):
                output[i] = func(x[i], *args)
            return output
    return vfunc