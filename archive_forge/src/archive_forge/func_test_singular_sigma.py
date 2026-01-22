from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
@pytest.mark.skip('Test does not raise but should')
def test_singular_sigma(self):
    n = len(self.endog)
    sigma = np.ones((n, n)) + np.diag(np.ones(n))
    sigma[0, 1] = sigma[1, 0] = 2
    assert np.linalg.matrix_rank(sigma) == n - 1
    with pytest.raises(np.linalg.LinAlgError):
        GLS(self.endog, self.exog, sigma=sigma)