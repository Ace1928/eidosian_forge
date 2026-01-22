from statsmodels.compat.platform import PLATFORM_WIN32
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises
from statsmodels.multivariate.pca import PCA, pca
from statsmodels.multivariate.tests.results.datamlw import (data, princomp1,
from statsmodels.tools.sm_exceptions import EstimationWarning
def test_equivalence_full_matrices(self):
    x = self.x.copy()
    svd_full_matrices_true = PCA(x, svd_full_matrices=True).factors
    svd_full_matrices_false = PCA(x).factors
    assert_allclose(svd_full_matrices_true, svd_full_matrices_false)