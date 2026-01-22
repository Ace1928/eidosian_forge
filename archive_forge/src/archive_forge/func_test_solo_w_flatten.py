import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_solo_w_flatten(self):
    w = self.data[0]
    test = merge_arrays(w, flatten=False)
    assert_equal(test, w)
    test = merge_arrays(w, flatten=True)
    control = np.array([(1, 2, 3.0), (4, 5, 6.0)], dtype=[('a', int), ('ba', float), ('bb', int)])
    assert_equal(test, control)