import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_raises)
from statsmodels.base.transform import (BoxCox)
from statsmodels.datasets import macrodata
def test_unclear_methods(self):
    assert_raises(ValueError, self.bc._est_lambda, self.x, (-1, 2), 'test')
    assert_raises(ValueError, self.bc.untransform_boxcox, self.x, 1, 'test')