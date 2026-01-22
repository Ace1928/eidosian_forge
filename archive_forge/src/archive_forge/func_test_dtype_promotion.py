import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_dtype_promotion(self):
    for mM in ['m', 'M']:
        assert_equal(np.promote_types(np.dtype(mM + '8[2Y]'), np.dtype(mM + '8[2Y]')), np.dtype(mM + '8[2Y]'))
        assert_equal(np.promote_types(np.dtype(mM + '8[12Y]'), np.dtype(mM + '8[15Y]')), np.dtype(mM + '8[3Y]'))
        assert_equal(np.promote_types(np.dtype(mM + '8[62M]'), np.dtype(mM + '8[24M]')), np.dtype(mM + '8[2M]'))
        assert_equal(np.promote_types(np.dtype(mM + '8[1W]'), np.dtype(mM + '8[2D]')), np.dtype(mM + '8[1D]'))
        assert_equal(np.promote_types(np.dtype(mM + '8[W]'), np.dtype(mM + '8[13s]')), np.dtype(mM + '8[s]'))
        assert_equal(np.promote_types(np.dtype(mM + '8[13W]'), np.dtype(mM + '8[49s]')), np.dtype(mM + '8[7s]'))
    assert_raises(TypeError, np.promote_types, np.dtype('m8[Y]'), np.dtype('m8[D]'))
    assert_raises(TypeError, np.promote_types, np.dtype('m8[M]'), np.dtype('m8[W]'))
    assert_raises(TypeError, np.promote_types, 'float32', 'm8')
    assert_raises(TypeError, np.promote_types, 'm8', 'float32')
    assert_raises(TypeError, np.promote_types, 'uint64', 'm8')
    assert_raises(TypeError, np.promote_types, 'm8', 'uint64')
    assert_raises(OverflowError, np.promote_types, np.dtype('m8[W]'), np.dtype('m8[fs]'))
    assert_raises(OverflowError, np.promote_types, np.dtype('m8[s]'), np.dtype('m8[as]'))