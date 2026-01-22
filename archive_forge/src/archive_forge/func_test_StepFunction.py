import numpy as np
import numpy.testing as npt
from numpy.testing import assert_raises
from statsmodels.distributions import StepFunction, monotone_fn_inverter
from statsmodels.distributions import ECDFDiscrete
def test_StepFunction(self):
    x = np.arange(20)
    y = np.arange(20)
    f = StepFunction(x, y)
    vals = f(np.array([[3.2, 4.5], [24, -3.1], [3.0, 4.0]]))
    npt.assert_almost_equal(vals, [[3, 4], [19, 0], [2, 3]])