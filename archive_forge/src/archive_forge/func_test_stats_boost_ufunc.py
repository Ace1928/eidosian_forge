import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import _boost
@pytest.mark.parametrize('func, args, expected', test_data)
def test_stats_boost_ufunc(func, args, expected):
    type_sigs = func.types
    type_chars = [sig.split('->')[-1] for sig in type_sigs]
    for type_char in type_chars:
        typ, rtol = type_char_to_type_tol[type_char]
        args = [typ(arg) for arg in args]
        with np.errstate(over='ignore'):
            value = func(*args)
        assert isinstance(value, typ)
        assert_allclose(value, expected, rtol=rtol)