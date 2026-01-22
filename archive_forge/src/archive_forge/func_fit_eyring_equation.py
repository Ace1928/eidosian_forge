import math
from .._util import get_backend
from ..util.pyutil import defaultnamedtuple
from ..units import default_units, Backend, default_constants, format_string
from .arrhenius import _get_R, _fit
def fit_eyring_equation(T, k, kerr=None, linearized=False, constants=None, units=None):
    """Curve fitting of the Eyring equation to data points

    Parameters
    ----------
    T : float
    k : array_like
    kerr : array_like (optional)
    linearized : bool

    """
    R = _get_R(constants, units)
    ln_kb_over_h = math.log(_get_kB_over_h(constants, units))
    return _fit(T, k, kerr, eyring_equation, lambda T, k: 1 / T, lambda T, k: np.log(k / T), [lambda p: -p[1] * R, lambda p: R * (p[0] - ln_kb_over_h)], linearized=linearized)