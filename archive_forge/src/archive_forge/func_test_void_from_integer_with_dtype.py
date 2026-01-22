import pytest
import numpy as np
from numpy.testing import (
@pytest.mark.parametrize('data', [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_from_integer_with_dtype(data):
    res = np.void(data, dtype='i,i')
    assert type(res) is np.void
    assert res.dtype == 'i,i'
    assert res['f0'] == 5 and res['f1'] == 5