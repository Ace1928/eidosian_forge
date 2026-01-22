from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
def fit_arrhenius_equation(T, k, kerr=None, linearized=False, constants=None, units=None):
    """Curve fitting of the Arrhenius equation to data points

    Parameters
    ----------
    T : float
    k : array_like
    kerr : array_like (optional)
    linearized : bool

    """
    return _fit(T, k, kerr, arrhenius_equation, lambda T, k: 1 / T, lambda T, k: np.log(k), [lambda p: np.exp(p[0]), lambda p: -p[1] * _get_R(constants, units)], linearized=linearized)