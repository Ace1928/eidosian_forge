import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_tests import get_buffer_info
import pytest
from numpy.testing import assert_, assert_equal, assert_raises
@pytest.mark.parametrize('scalar, code', scalars_and_codes, ids=codes_only)
def test_scalar_code_and_properties(self, scalar, code):
    x = scalar()
    expected = dict(strides=(), itemsize=x.dtype.itemsize, ndim=0, shape=(), format=code, readonly=True)
    mv_x = memoryview(x)
    assert self._as_dict(mv_x) == expected