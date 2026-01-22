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
def test_input_size_jl_min_dim():
    with pytest.raises(ValueError):
        johnson_lindenstrauss_min_dim(3 * [100], eps=2 * [0.9])
    johnson_lindenstrauss_min_dim(np.random.randint(1, 10, size=(10, 10)), eps=np.full((10, 10), 0.5))