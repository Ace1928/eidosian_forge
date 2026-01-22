from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
def limiting_log_gamma(IS, z, A, I0=1, backend=None):
    """Debye-Hyckel limiting formula"""
    be = get_backend(backend)
    one = be.pi ** 0
    return -A * z ** 2 * (IS / I0) ** (one / 2)