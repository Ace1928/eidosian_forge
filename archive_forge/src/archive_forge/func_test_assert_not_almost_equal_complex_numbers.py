import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('a,b,rtol', [(0.58310768, 0.58330768, 1e-07), (-0.908 + 0.2j, -0.978 + 0.2j, 0.001), (0.1 + 1j, 0.1 + 2j, 0.01), (-0.132 + 1.001j, -0.132 + 1.005j, 1e-05), (0.58310768j, 0.58330768j, 1e-09)])
def test_assert_not_almost_equal_complex_numbers(a, b, rtol):
    _assert_not_almost_equal_both(a, b, rtol=rtol)
    _assert_not_almost_equal_both(np.complex64(a), np.complex64(b), rtol=rtol)
    _assert_not_almost_equal_both(np.complex128(a), np.complex128(b), rtol=rtol)