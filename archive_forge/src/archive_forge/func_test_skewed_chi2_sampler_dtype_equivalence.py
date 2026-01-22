import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_skewed_chi2_sampler_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    skewed_chi2_sampler_32 = SkewedChi2Sampler(random_state=42)
    X_32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    skewed_chi2_sampler_32.fit(X_32)
    skewed_chi2_sampler_64 = SkewedChi2Sampler(random_state=42)
    X_64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    skewed_chi2_sampler_64.fit(X_64)
    assert_allclose(skewed_chi2_sampler_32.random_offset_, skewed_chi2_sampler_64.random_offset_)
    assert_allclose(skewed_chi2_sampler_32.random_weights_, skewed_chi2_sampler_64.random_weights_)