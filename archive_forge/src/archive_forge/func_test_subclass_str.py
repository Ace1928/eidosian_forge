import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_subclass_str(self):
    """test str with subclass that has overridden str, setitem"""
    x = np.arange(5)
    xsub = SubArray(x)
    mxsub = masked_array(xsub, mask=[True, False, True, False, False])
    assert_equal(str(mxsub), '[-- 1 -- 3 4]')
    xcsub = ComplicatedSubArray(x)
    assert_raises(ValueError, xcsub.__setitem__, 0, np.ma.core.masked_print_option)
    mxcsub = masked_array(xcsub, mask=[True, False, True, False, False])
    assert_equal(str(mxcsub), 'myprefix [-- 1 -- 3 4] mypostfix')