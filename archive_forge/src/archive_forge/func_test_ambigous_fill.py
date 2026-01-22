import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_ambigous_fill(self):
    a = np.array([[3, 3, 255], [3, 3, 255]], dtype=np.uint8)
    a = np.ma.masked_array(a, mask=a == 3)
    assert_array_equal(np.ma.median(a, axis=1), 255)
    assert_array_equal(np.ma.median(a, axis=1).mask, False)
    assert_array_equal(np.ma.median(a, axis=0), a[0])
    assert_array_equal(np.ma.median(a), 255)