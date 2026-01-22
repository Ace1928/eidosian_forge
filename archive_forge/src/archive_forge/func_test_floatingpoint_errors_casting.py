import pytest
from pytest import param
from numpy.testing import IS_WASM
import numpy as np
@pytest.mark.skipif(IS_WASM, reason='no wasm fp exception support')
@pytest.mark.parametrize(['value', 'dtype'], values_and_dtypes())
@pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
def test_floatingpoint_errors_casting(dtype, value):
    dtype = np.dtype(dtype)
    for operation in check_operations(dtype, value):
        dtype = np.dtype(dtype)
        match = 'invalid' if dtype.kind in 'iu' else 'overflow'
        with pytest.warns(RuntimeWarning, match=match):
            operation()
        with np.errstate(all='raise'):
            with pytest.raises(FloatingPointError, match=match):
                operation()