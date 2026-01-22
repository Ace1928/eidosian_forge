from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
def extended_activity_product(IS, stoich, z, a, T, eps_r, rho, C=0, backend=None):
    be = get_backend(backend)
    Aval = A(eps_r, T, rho)
    Bval = B(eps_r, T, rho)
    tot = 0
    for idx, nr in enumerate(stoich):
        tot += nr * extended_log_gamma(IS, z[idx], a[idx], Aval, Bval, C)
    return be.exp(tot)