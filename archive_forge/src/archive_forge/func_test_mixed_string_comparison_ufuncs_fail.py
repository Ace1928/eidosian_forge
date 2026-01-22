import pytest
import operator
import numpy as np
from numpy.testing import assert_array_equal
@pytest.mark.parametrize(['op', 'ufunc', 'sym'], COMPARISONS)
def test_mixed_string_comparison_ufuncs_fail(op, ufunc, sym):
    arr_string = np.array(['a', 'b'], dtype='S')
    arr_unicode = np.array(['a', 'c'], dtype='U')
    with pytest.raises(TypeError, match='did not contain a loop'):
        ufunc(arr_string, arr_unicode)
    with pytest.raises(TypeError, match='did not contain a loop'):
        ufunc(arr_unicode, arr_string)