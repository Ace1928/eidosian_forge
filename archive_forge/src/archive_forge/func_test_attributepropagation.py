import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_attributepropagation(self):
    x = array(arange(5), mask=[0] + [1] * 4)
    my = masked_array(subarray(x))
    ym = msubarray(x)
    z = my + 1
    assert_(isinstance(z, MaskedArray))
    assert_(not isinstance(z, MSubArray))
    assert_(isinstance(z._data, SubArray))
    assert_equal(z._data.info, {})
    z = ym + 1
    assert_(isinstance(z, MaskedArray))
    assert_(isinstance(z, MSubArray))
    assert_(isinstance(z._data, SubArray))
    assert_(z._data.info['added'] > 0)
    ym += 1
    assert_(isinstance(ym, MaskedArray))
    assert_(isinstance(ym, MSubArray))
    assert_(isinstance(ym._data, SubArray))
    assert_(ym._data.info['iadded'] > 0)
    ym._set_mask([1, 0, 0, 0, 1])
    assert_equal(ym._mask, [1, 0, 0, 0, 1])
    ym._series._set_mask([0, 0, 0, 0, 1])
    assert_equal(ym._mask, [0, 0, 0, 0, 1])
    xsub = subarray(x, info={'name': 'x'})
    mxsub = masked_array(xsub)
    assert_(hasattr(mxsub, 'info'))
    assert_equal(mxsub.info, xsub.info)