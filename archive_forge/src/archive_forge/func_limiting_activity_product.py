from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
def limiting_activity_product(IS, stoich, z, T, eps_r, rho, backend=None):
    """Product of activity coefficients based on DH limiting law."""
    be = get_backend(backend)
    Aval = A(eps_r, T, rho)
    tot = 0
    for idx, nr in enumerate(stoich):
        tot += nr * limiting_log_gamma(IS, z[idx], Aval)
    return be.exp(tot)