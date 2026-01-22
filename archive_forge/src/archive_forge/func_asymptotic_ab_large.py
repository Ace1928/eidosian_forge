import warnings
from collections.abc import Iterable
from functools import wraps, cached_property
import ctypes
import numpy as np
from numpy.polynomial import Polynomial
from scipy._lib.doccer import (extend_notes_in_docstring,
from scipy._lib._ccallback import LowLevelCallable
from scipy import optimize
from scipy import integrate
import scipy.special as sc
import scipy.special._ufuncs as scu
from scipy._lib._util import _lazyselect, _lazywhere
from . import _stats
from ._tukeylambda_stats import (tukeylambda_variance as _tlvar,
from ._distn_infrastructure import (
from ._ksstats import kolmogn, kolmognp, kolmogni
from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
from ._censored_data import CensoredData
import scipy.stats._boost as _boost
from scipy.optimize import root_scalar
from scipy.stats._warnings_errors import FitError
import scipy.stats as stats
def asymptotic_ab_large(a, b):
    sum_ab = a + b
    log_term = 0.5 * (np.log(2 * np.pi) + np.log(a) + np.log(b) - 3 * np.log(sum_ab) + 1)
    t1 = 110 / sum_ab + 20 * sum_ab ** (-2.0) + sum_ab ** (-3.0) - 2 * sum_ab ** (-4.0)
    t2 = -50 / a - 10 * a ** (-2.0) - a ** (-3.0) + a ** (-4.0)
    t3 = -50 / b - 10 * b ** (-2.0) - b ** (-3.0) + b ** (-4.0)
    return log_term + (t1 + t2 + t3) / 120