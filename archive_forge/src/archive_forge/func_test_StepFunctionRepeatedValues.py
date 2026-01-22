import numpy as np
import numpy.testing as npt
from numpy.testing import assert_raises
from statsmodels.distributions import StepFunction, monotone_fn_inverter
from statsmodels.distributions import ECDFDiscrete
def test_StepFunctionRepeatedValues(self):
    x = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5]
    y = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    f = StepFunction(x, y)
    npt.assert_almost_equal(f([1, 2, 3, 4, 5]), [0, 7, 10, 13, 14])
    f2 = StepFunction(x, y, side='right')
    npt.assert_almost_equal(f2([1, 2, 3, 4, 5]), [7, 10, 13, 14, 15])