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
def check_input_size_random_matrix(random_matrix):
    inputs = [(0, 0), (-1, 1), (1, -1), (1, 0), (-1, 0)]
    for n_components, n_features in inputs:
        with pytest.raises(ValueError):
            random_matrix(n_components, n_features)