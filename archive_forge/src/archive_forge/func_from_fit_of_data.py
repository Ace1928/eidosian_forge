from .._util import get_backend
from ..util.regression import least_squares
from ..util.pyutil import defaultnamedtuple
from ..units import default_constants, default_units, format_string, patched_numpy
@classmethod
def from_fit_of_data(cls, T, k, kerr=None, **kwargs):
    args, vcv = fit_arrhenius_equation(T, k, kerr)
    return cls(*args, **kwargs)