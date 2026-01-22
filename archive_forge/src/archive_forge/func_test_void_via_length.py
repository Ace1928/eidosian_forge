import pytest
import numpy as np
from numpy.testing import (
@pytest.mark.parametrize('length', [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_via_length(length):
    res = np.void(length)
    assert type(res) is np.void
    assert res.item() == b'\x00' * 5
    assert res.dtype == 'V5'