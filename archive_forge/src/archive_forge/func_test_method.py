import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import pytest
from scipy import stats
from scipy.stats import sobol_indices
from scipy.stats._resampling import BootstrapResult
from scipy.stats._sensitivity_analysis import (
def test_method(self, ishigami_ref_indices):

    def jansen_sobol(f_A, f_B, f_AB):
        """Jansen for S and Sobol' for St.

            From Saltelli2010, table 2 formulations (c) and (e)."""
        var = np.var([f_A, f_B], axis=(0, -1))
        s = (var - 0.5 * np.mean((f_B - f_AB) ** 2, axis=-1)) / var
        st = np.mean(f_A * (f_A - f_AB), axis=-1) / var
        return (s.T, st.T)
    rng = np.random.default_rng(28631265345463262246170309650372465332)
    res = sobol_indices(func=f_ishigami, n=4096, dists=self.dists, method=jansen_sobol, random_state=rng)
    assert_allclose(res.first_order, ishigami_ref_indices[0], atol=0.01)
    assert_allclose(res.total_order, ishigami_ref_indices[1], atol=0.01)

    def jansen_sobol_typed(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return jansen_sobol(f_A, f_B, f_AB)
    _ = sobol_indices(func=f_ishigami, n=8, dists=self.dists, method=jansen_sobol_typed, random_state=rng)