import pytest
import operator
import numpy as np
from numpy.testing import assert_array_equal
@pytest.mark.parametrize(['op', 'ufunc', 'sym'], COMPARISONS)
def test_mixed_string_comparisons_ufuncs_with_cast(op, ufunc, sym):
    arr_string = np.array(['a', 'b'], dtype='S')
    arr_unicode = np.array(['a', 'c'], dtype='U')
    res1 = ufunc(arr_string, arr_unicode, signature='UU->?', casting='unsafe')
    res2 = ufunc(arr_string, arr_unicode, signature='SS->?', casting='unsafe')
    expected = op(arr_string.astype('U'), arr_unicode)
    assert_array_equal(res1, expected)
    assert_array_equal(res2, expected)