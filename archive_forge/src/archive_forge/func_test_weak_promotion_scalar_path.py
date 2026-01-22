import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
@pytest.mark.parametrize('op', [operator.add, operator.pow, operator.eq])
def test_weak_promotion_scalar_path(op):
    np._set_promotion_state('weak')
    res = op(np.uint8(3), 5)
    assert res == op(3, 5)
    assert res.dtype == np.uint8 or res.dtype == bool
    with pytest.raises(OverflowError):
        op(np.uint8(3), 1000)
    res = op(np.float32(3), 5.0)
    assert res == op(3.0, 5.0)
    assert res.dtype == np.float32 or res.dtype == bool