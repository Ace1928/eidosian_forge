import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
@pytest.mark.parametrize('ufunc', [np.add, np.power])
@pytest.mark.parametrize('state', ['weak', 'weak_and_warn'])
def test_nep50_huge_integers(ufunc, state):
    np._set_promotion_state(state)
    with pytest.raises(OverflowError):
        ufunc(np.int64(0), 2 ** 63)
    if state == 'weak_and_warn':
        with pytest.warns(UserWarning, match='result dtype changed.*float64.*uint64'):
            with pytest.raises(OverflowError):
                ufunc(np.uint64(0), 2 ** 64)
    else:
        with pytest.raises(OverflowError):
            ufunc(np.uint64(0), 2 ** 64)
    if state == 'weak_and_warn':
        with pytest.warns(UserWarning, match='result dtype changed.*float64.*uint64'):
            res = ufunc(np.uint64(1), 2 ** 63)
    else:
        res = ufunc(np.uint64(1), 2 ** 63)
    assert res.dtype == np.uint64
    assert res == ufunc(1, 2 ** 63, dtype=object)
    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2 ** 63)
    with pytest.raises(OverflowError):
        ufunc(np.int64(1), 2 ** 100)
    res = ufunc(1.0, 2 ** 100)
    assert isinstance(res, np.float64)