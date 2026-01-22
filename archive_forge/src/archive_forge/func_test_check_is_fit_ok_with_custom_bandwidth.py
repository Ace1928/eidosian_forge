import os
import numpy.testing as npt
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.kde import KDEUnivariate as KDE
import statsmodels.sandbox.nonparametric.kernels as kernels
import statsmodels.nonparametric.bandwidths as bandwidths
def test_check_is_fit_ok_with_custom_bandwidth(self):

    def custom_bw(X, kern):
        return np.std(X) * len(X)
    kde = self.kde.fit(bw=custom_bw)
    assert isinstance(kde, KDE)