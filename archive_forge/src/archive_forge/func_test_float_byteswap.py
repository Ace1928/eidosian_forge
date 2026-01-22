from hypothesis import (
import numpy as np
import pytest
from pandas._libs.byteswap import (
import pandas._testing as tm
@pytest.mark.filterwarnings('ignore:overflow encountered:RuntimeWarning')
@given(read_offset=st.integers(0, 11), number=st.floats())
@pytest.mark.parametrize('float_type', [np.float32, np.float64])
@pytest.mark.parametrize('should_byteswap', [True, False])
def test_float_byteswap(read_offset, number, float_type, should_byteswap):
    _test(number, float_type, read_offset, should_byteswap)