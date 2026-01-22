import math
from .._util import get_backend
from ..util.pyutil import defaultnamedtuple
from ..units import default_units, Backend, default_constants, format_string
from .arrhenius import _get_R, _fit
def kB_h_times_exp_dS_R(self, constants=None, units=None, backend=math):
    R = _get_R(constants, units)
    kB_over_h = _get_kB_over_h(constants, units)
    return kB_over_h * backend.exp(self.dS / R)