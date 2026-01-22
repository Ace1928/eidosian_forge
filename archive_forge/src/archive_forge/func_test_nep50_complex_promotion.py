import operator
import numpy as np
import pytest
from numpy.testing import IS_WASM
def test_nep50_complex_promotion():
    np._set_promotion_state('weak')
    with pytest.warns(RuntimeWarning, match='.*overflow'):
        res = np.complex64(3) + complex(2 ** 300)
    assert type(res) == np.complex64