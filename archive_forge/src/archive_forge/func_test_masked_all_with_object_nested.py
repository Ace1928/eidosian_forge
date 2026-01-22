import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_masked_all_with_object_nested(self):
    my_dtype = np.dtype([('b', ([('c', object)], (1,)))])
    masked_arr = np.ma.masked_all((1,), my_dtype)
    assert_equal(type(masked_arr['b']), np.ma.core.MaskedArray)
    assert_equal(type(masked_arr['b']['c']), np.ma.core.MaskedArray)
    assert_equal(len(masked_arr['b']['c']), 1)
    assert_equal(masked_arr['b']['c'].shape, (1, 1))
    assert_equal(masked_arr['b']['c']._fill_value.shape, ())