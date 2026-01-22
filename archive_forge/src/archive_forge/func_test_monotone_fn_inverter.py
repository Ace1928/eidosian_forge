import numpy as np
import numpy.testing as npt
from numpy.testing import assert_raises
from statsmodels.distributions import StepFunction, monotone_fn_inverter
from statsmodels.distributions import ECDFDiscrete
def test_monotone_fn_inverter(self):
    x = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    fn = lambda x: 1.0 / x
    y = fn(np.array(x))
    f = monotone_fn_inverter(fn, x)
    npt.assert_array_equal(f.y, x[::-1])
    npt.assert_array_equal(f.x, y[::-1])