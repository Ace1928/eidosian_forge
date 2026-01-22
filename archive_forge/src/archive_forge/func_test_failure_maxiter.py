from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('locally_biased', [True, False])
def test_failure_maxiter(self, locally_biased):
    maxiter = 10
    result = direct(self.styblinski_tang, self.bounds_stylinski_tang, maxiter=maxiter, locally_biased=locally_biased)
    assert result.success is False
    assert result.status == 2
    assert result.nit >= maxiter
    message = f'Number of iterations is larger than maxiter={maxiter}'
    assert result.message == message