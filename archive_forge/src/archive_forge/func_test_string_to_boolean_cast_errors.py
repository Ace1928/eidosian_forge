import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
@pytest.mark.parametrize(['dtype', 'out_dtype'], [(np.bytes_, np.bool_), (np.str_, np.bool_), (np.dtype('S10,S9'), np.dtype('?,?'))])
def test_string_to_boolean_cast_errors(dtype, out_dtype):
    """
    These currently error out, since cast to integers fails, but should not
    error out in the future.
    """
    for invalid in ['False', 'True', '', '\x00', 'non-empty']:
        arr = np.array([invalid], dtype=dtype)
        with assert_raises(ValueError):
            arr.astype(out_dtype)