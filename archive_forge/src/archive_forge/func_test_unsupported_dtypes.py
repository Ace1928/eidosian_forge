import itertools
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
from pytest import raises as assert_raises
from scipy import linalg
import scipy.linalg._decomp_update as _decomp_update
from scipy.linalg._decomp_update import qr_delete, qr_update, qr_insert
def test_unsupported_dtypes(self):
    dts = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'longdouble', 'clongdouble', 'bool']
    a, q0, r0, u0, v0 = self.generate('tall')
    for dtype in dts:
        q = q0.real.astype(dtype)
        with np.errstate(invalid='ignore'):
            r = r0.real.astype(dtype)
        u = u0.real.astype(dtype)
        v = v0.real.astype(dtype)
        assert_raises(ValueError, qr_update, q, r0, u0, v0)
        assert_raises(ValueError, qr_update, q0, r, u0, v0)
        assert_raises(ValueError, qr_update, q0, r0, u, v0)
        assert_raises(ValueError, qr_update, q0, r0, u0, v)