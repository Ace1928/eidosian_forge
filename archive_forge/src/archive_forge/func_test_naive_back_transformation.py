import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
def test_naive_back_transformation(self):
    y_zero_lambda = self.bc.transform_boxcox(self.x, 0.0)
    y_half_lambda = self.bc.transform_boxcox(self.x, 0.5)
    y_zero_lambda_un = self.bc.untransform_boxcox(*y_zero_lambda, method='naive')
    y_half_lambda_un = self.bc.untransform_boxcox(*y_half_lambda, method='naive')
    assert_almost_equal(self.x, y_zero_lambda_un, 5)
    assert_almost_equal(self.x, y_half_lambda_un, 5)