import numpy as np
from scipy import stats
from statsmodels.sandbox.nonparametric import kernels
from statsmodels.distributions.mixture_rvs import mixture_rvs
from statsmodels.nonparametric.bandwidths import select_bandwidth
from statsmodels.nonparametric.bandwidths import bw_normal_reference
from numpy.testing import assert_allclose
import pytest
def test_bandwidth_zero(self):
    kern = kernels.Gaussian()
    for bw in ['scott', 'silverman', 'normal_reference']:
        with pytest.raises(RuntimeError, match='Selected KDE bandwidth is 0'):
            select_bandwidth(self.xx, bw, kern)