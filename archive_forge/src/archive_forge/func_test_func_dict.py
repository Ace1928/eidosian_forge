import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
def test_func_dict(self, ishigami_ref_indices):
    rng = np.random.default_rng(28631265345463262246170309650372465332)
    n = 4096
    dists = [stats.uniform(loc=-np.pi, scale=2 * np.pi), stats.uniform(loc=-np.pi, scale=2 * np.pi), stats.uniform(loc=-np.pi, scale=2 * np.pi)]
    A, B = sample_A_B(n=n, dists=dists, random_state=rng)
    AB = sample_AB(A=A, B=B)
    func = {'f_A': f_ishigami(A).reshape(1, -1), 'f_B': f_ishigami(B).reshape(1, -1), 'f_AB': f_ishigami(AB).reshape((3, 1, -1))}
    res = sobol_indices(func=func, n=n, dists=dists, random_state=rng)
    assert_allclose(res.first_order, ishigami_ref_indices[0], atol=0.01)
    res = sobol_indices(func=func, n=n, random_state=rng)
    assert_allclose(res.first_order, ishigami_ref_indices[0], atol=0.01)