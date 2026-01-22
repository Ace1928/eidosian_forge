import numpy.testing as npt
from scipy.optimize.cython_optimize import _zeros
def test_brentq_full_output():
    output = _zeros.full_output_example((A0[0],) + ARGS, XLO, XHI, XTOL, RTOL, MITR)
    npt.assert_allclose(EXPECTED[0], output['root'], rtol=RTOL, atol=XTOL)
    npt.assert_equal(6, output['iterations'])
    npt.assert_equal(7, output['funcalls'])
    npt.assert_equal(0, output['error_num'])