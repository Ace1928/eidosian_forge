import warnings
from numpy import zeros, concatenate, ravel, diff, array, ones  # noqa:F401
import numpy as np
from . import _fitpack_impl
from . import dfitpack
def set_smoothing_factor(self, s):
    """ Continue spline computation with the given smoothing
        factor s and with the knots found at the last call.

        This routine modifies the spline in place.

        """
    data = self._data
    if data[6] == -1:
        warnings.warn('smoothing factor unchanged forLSQ spline with fixed knots', stacklevel=2)
        return
    args = data[:6] + (s,) + data[7:]
    data = dfitpack.fpcurf1(*args)
    if data[-1] == 1:
        data = self._reset_nest(data)
    self._data = data
    self._reset_class()