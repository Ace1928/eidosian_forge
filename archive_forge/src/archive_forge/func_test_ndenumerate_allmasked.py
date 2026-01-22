import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_ndenumerate_allmasked(self):
    a = masked_all(())
    b = masked_all((100,))
    c = masked_all((2, 3, 4))
    assert_equal(list(ndenumerate(a)), [])
    assert_equal(list(ndenumerate(b)), [])
    assert_equal(list(ndenumerate(b, compressed=False)), list(zip(np.ndindex((100,)), 100 * [masked])))
    assert_equal(list(ndenumerate(c)), [])
    assert_equal(list(ndenumerate(c, compressed=False)), list(zip(np.ndindex((2, 3, 4)), 2 * 3 * 4 * [masked])))