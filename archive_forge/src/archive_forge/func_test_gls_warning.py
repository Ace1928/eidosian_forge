from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_gls_warning(reset_randomstate):
    data = np.random.standard_normal((400, 200))
    data[:, 1:] = data[:, :1] + 0.01 * data[:, 1:]
    with pytest.warns(EstimationWarning, match='Many series are being down weighted'):
        factors = PCA(data, ncomp=2, gls=True).factors
    assert factors.shape == (data.shape[0], 2)