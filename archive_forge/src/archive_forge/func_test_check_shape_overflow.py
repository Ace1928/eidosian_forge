import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix
def test_check_shape_overflow(self):
    new_shape = sputils.check_shape([(10, -1)], (65535, 131070))
    assert_equal(new_shape, (10, 858967245))