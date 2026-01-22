from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.xslow
@pytest.mark.parametrize('locally_biased', [True, False])
def test_no_segmentation_fault(self, locally_biased):
    bounds = [(-5.0, 20.0)] * 100
    result = direct(self.sphere, bounds, maxfun=10000000, maxiter=1000000, locally_biased=locally_biased)
    assert result is not None