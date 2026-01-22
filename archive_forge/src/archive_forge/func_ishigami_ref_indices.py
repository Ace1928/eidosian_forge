import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
@pytest.fixture(scope='session')
def ishigami_ref_indices():
    """Reference values for Ishigami from Saltelli2007.

    Chapter 4, exercise 5 pages 179-182.
    """
    a = 7.0
    b = 0.1
    var = 0.5 + a ** 2 / 8 + b * np.pi ** 4 / 5 + b ** 2 * np.pi ** 8 / 18
    v1 = 0.5 + b * np.pi ** 4 / 5 + b ** 2 * np.pi ** 8 / 50
    v2 = a ** 2 / 8
    v3 = 0
    v12 = 0
    v13 = b ** 2 * np.pi ** 8 * 8 / 225
    v23 = 0
    s_first = np.array([v1, v2, v3]) / var
    s_second = np.array([[0.0, 0.0, v13], [v12, 0.0, v23], [v13, v23, 0.0]]) / var
    s_total = s_first + s_second.sum(axis=1)
    return (s_first, s_total)