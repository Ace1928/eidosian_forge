from collections import OrderedDict
import warnings
from ._util import get_backend
from .chemistry import Substance
from .units import allclose
def davies_log_gamma(IS, z, A, C=-0.3, I0=1, backend=None):
    """Davies formula"""
    be = get_backend(backend)
    one = be.pi ** 0
    I_I0 = IS / I0
    sqrt_I_I0 = I_I0 ** (one / 2)
    return -A * z ** 2 * (sqrt_I_I0 / (1 + sqrt_I_I0) + C * I_I0)