import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_nystroem_callable():
    rnd = np.random.RandomState(42)
    n_samples = 10
    X = rnd.uniform(size=(n_samples, 4))

    def logging_histogram_kernel(x, y, log):
        """Histogram kernel that writes to a log."""
        log.append(1)
        return np.minimum(x, y).sum()
    kernel_log = []
    X = list(X)
    Nystroem(kernel=logging_histogram_kernel, n_components=n_samples - 1, kernel_params={'log': kernel_log}).fit(X)
    assert len(kernel_log) == n_samples * (n_samples - 1) / 2
    msg = "Don't pass gamma, coef0 or degree to Nystroem"
    params = ({'gamma': 1}, {'coef0': 1}, {'degree': 2})
    for param in params:
        ny = Nystroem(kernel=_linear_kernel, n_components=n_samples - 1, **param)
        with pytest.raises(ValueError, match=msg):
            ny.fit(X)