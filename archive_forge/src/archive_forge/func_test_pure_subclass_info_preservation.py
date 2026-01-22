import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_pure_subclass_info_preservation(self):
    arr1 = SubMaskedArray('test', data=[1, 2, 3, 4, 5, 6])
    arr2 = SubMaskedArray(data=[0, 1, 2, 3, 4, 5])
    diff1 = np.subtract(arr1, arr2)
    assert_('info' in diff1._optinfo)
    assert_(diff1._optinfo['info'] == 'test')
    diff2 = arr1 - arr2
    assert_('info' in diff2._optinfo)
    assert_(diff2._optinfo['info'] == 'test')