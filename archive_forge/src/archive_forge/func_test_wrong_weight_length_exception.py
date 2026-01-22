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
def test_wrong_weight_length_exception(self):
    with pytest.raises(ValueError):
        self.kde.fit(kernel='gau', gridsize=50, weights=self.weights_100, fft=False, bw='silverman')