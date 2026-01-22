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
def simps(y, x=None, dx=1.0, axis=-1, even=_NoValue):
    """An alias of `simpson`.

    `simps` is kept for backwards compatibility. For new code, prefer
    `simpson` instead.
    """
    msg = "'scipy.integrate.simps' is deprecated in favour of 'scipy.integrate.simpson' and will be removed in SciPy 1.14.0"
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return simpson(y, x=x, dx=dx, axis=axis, even=even)