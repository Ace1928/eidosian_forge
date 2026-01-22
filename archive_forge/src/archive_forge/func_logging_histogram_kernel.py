import re
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.kernel_approximation import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def logging_histogram_kernel(x, y, log):
    """Histogram kernel that writes to a log."""
    log.append(1)
    return np.minimum(x, y).sum()