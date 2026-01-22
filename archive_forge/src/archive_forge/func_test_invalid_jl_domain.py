import functools
import warnings
from typing import Any, List
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.random_projection import (
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS
@pytest.mark.parametrize('n_samples, eps', [([100, 110], [0.9, 1.1]), ([90, 100], [0.1, 0.0]), ([50, -40], [0.1, 0.2])])
def test_invalid_jl_domain(n_samples, eps):
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(n_samples, eps=eps)