import pytest
from numpy.testing import assert_allclose
from scipy.special._test_internal import _dd_exp, _dd_log, _dd_expm1
@pytest.mark.parametrize('dd_func, xhi, xlo, expected_yhi, expected_ylo', test_data)
def test_dd(dd_func, xhi, xlo, expected_yhi, expected_ylo):
    yhi, ylo = dd_func(xhi, xlo)
    assert yhi == expected_yhi, f'high double ({yhi}) does not equal the expected value {expected_yhi}'
    assert_allclose(ylo, expected_ylo, rtol=5e-15)