import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_rbf_sampler_dtype_equivalence():
    """Check the equivalence of the results with 32 and 64 bits input."""
    rbf32 = RBFSampler(random_state=42)
    X32 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    rbf32.fit(X32)
    rbf64 = RBFSampler(random_state=42)
    X64 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    rbf64.fit(X64)
    assert_allclose(rbf32.random_offset_, rbf64.random_offset_)
    assert_allclose(rbf32.random_weights_, rbf64.random_weights_)