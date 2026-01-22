import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
@pytest.mark.parametrize('dtype', np.typecodes['AllInteger'])
def test_nep50_weak_integers(dtype):
    np._set_promotion_state('weak')
    scalar_type = np.dtype(dtype).type
    maxint = int(np.iinfo(dtype).max)
    with np.errstate(over='warn'):
        with pytest.warns(RuntimeWarning):
            res = scalar_type(100) + maxint
    assert res.dtype == dtype
    res = np.array(100, dtype=dtype) + maxint
    assert res.dtype == dtype