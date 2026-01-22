import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
@pytest.mark.parametrize('dtype', np.typecodes['AllFloat'])
def test_nep50_weak_integers_with_inexact(dtype):
    np._set_promotion_state('weak')
    scalar_type = np.dtype(dtype).type
    too_big_int = int(np.finfo(dtype).max) * 2
    if dtype in 'dDG':
        with pytest.raises(OverflowError):
            scalar_type(1) + too_big_int
        with pytest.raises(OverflowError):
            np.array(1, dtype=dtype) + too_big_int
    else:
        if dtype in 'gG':
            try:
                str(too_big_int)
            except ValueError:
                pytest.skip('`huge_int -> string -> longdouble` failed')
        with pytest.warns(RuntimeWarning):
            res = scalar_type(1) + too_big_int
        assert res.dtype == dtype
        assert res == np.inf
        with pytest.warns(RuntimeWarning):
            res = np.add(np.array(1, dtype=dtype), too_big_int, dtype=dtype)
        assert res.dtype == dtype
        assert res == np.inf