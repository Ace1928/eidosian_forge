import warnings
import numpy as np
import pytest
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import assert_equal, assert_, assert_raises
from numpy.core.getlimits import _discovered_machar, _float_ma
def test_known_types():
    for ftype, ma_like in ((np.float16, _float_ma[16]), (np.float32, _float_ma[32]), (np.float64, _float_ma[64])):
        assert_ma_equal(_discovered_machar(ftype), ma_like)
    with np.errstate(all='ignore'):
        ld_ma = _discovered_machar(np.longdouble)
    bytes = np.dtype(np.longdouble).itemsize
    if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
        assert_ma_equal(ld_ma, _float_ma[80])
    elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
        assert_ma_equal(ld_ma, _float_ma[128])