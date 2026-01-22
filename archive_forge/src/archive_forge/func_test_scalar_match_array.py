import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import get_buffer_info
import pytest
from numpy.testing import assert_, assert_equal, assert_raises
@pytest.mark.parametrize('scalar', scalars_only, ids=codes_only)
def test_scalar_match_array(self, scalar):
    x = scalar()
    a = np.array([], dtype=np.dtype(scalar))
    mv_x = memoryview(x)
    mv_a = memoryview(a)
    assert_equal(mv_x.format, mv_a.format)