import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
def test_timedelta_modulus_div_by_zero(self):
    with assert_warns(RuntimeWarning):
        actual = np.timedelta64(10, 's') % np.timedelta64(0, 's')
        assert_equal(actual, np.timedelta64('NaT'))