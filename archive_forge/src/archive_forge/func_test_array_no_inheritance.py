import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_array_no_inheritance():
    data_masked = np.ma.array([1, 2, 3], mask=[True, False, True])
    data_masked_units = ArrayNoInheritance(data_masked, 'meters')
    new_array = np.ma.array(data_masked_units)
    assert_equal(data_masked.data, new_array.data)
    assert_equal(data_masked.mask, new_array.mask)
    data_masked.mask = [True, False, False]
    assert_equal(data_masked.mask, new_array.mask)
    assert_(new_array.sharedmask)
    new_array = np.ma.array(data_masked_units, copy=True)
    assert_equal(data_masked.data, new_array.data)
    assert_equal(data_masked.mask, new_array.mask)
    data_masked.mask = [True, False, True]
    assert_equal([True, False, False], new_array.mask)
    assert_(not new_array.sharedmask)
    new_array = np.ma.array(data_masked_units, keep_mask=False)
    assert_equal(data_masked.data, new_array.data)
    assert_equal(data_masked.mask, [True, False, True])
    assert_(not new_array.mask)
    assert_(not new_array.sharedmask)