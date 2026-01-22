import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
def test_subclass_nomask_items(self):
    x = np.arange(5)
    xcsub = ComplicatedSubArray(x)
    mxcsub_nomask = masked_array(xcsub)
    assert_(isinstance(mxcsub_nomask[1, ...].data, ComplicatedSubArray))
    assert_(isinstance(mxcsub_nomask[0, ...].data, ComplicatedSubArray))
    assert_(isinstance(mxcsub_nomask[1], ComplicatedSubArray))
    assert_(isinstance(mxcsub_nomask[0], ComplicatedSubArray))