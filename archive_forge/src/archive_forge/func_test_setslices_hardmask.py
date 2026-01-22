import numpy as np
import numpy.ma as ma
from numpy import recarray
from numpy.ma import masked, nomask
from numpy.testing import temppath
from numpy.core.records import (
from numpy.ma.mrecords import (
from numpy.ma.testutils import (
from numpy.compat import pickle
def test_setslices_hardmask(self):
    base = self.base.copy()
    mbase = base.view(mrecarray)
    mbase.harden_mask()
    try:
        mbase[-2:] = (5, 5, 5)
        assert_equal(mbase.a._data, [1, 2, 3, 5, 5])
        assert_equal(mbase.b._data, [1.1, 2.2, 3.3, 5, 5.5])
        assert_equal(mbase.c._data, [b'one', b'two', b'three', b'5', b'five'])
        assert_equal(mbase.a._mask, [0, 1, 0, 0, 1])
        assert_equal(mbase.b._mask, mbase.a._mask)
        assert_equal(mbase.b._mask, mbase.c._mask)
    except NotImplementedError:
        pass
    except AssertionError:
        raise
    else:
        raise Exception('Flexible hard masks should be supported !')
    try:
        mbase[-2:] = 3
    except (NotImplementedError, TypeError):
        pass
    else:
        raise TypeError('Should have expected a readable buffer object!')