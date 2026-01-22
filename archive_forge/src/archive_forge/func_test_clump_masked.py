import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_clump_masked(self):
    a = masked_array(np.arange(10))
    a[[0, 1, 2, 6, 8, 9]] = masked
    test = clump_masked(a)
    control = [slice(0, 3), slice(6, 7), slice(8, 10)]
    assert_equal(test, control)
    self.check_clump(clump_masked)