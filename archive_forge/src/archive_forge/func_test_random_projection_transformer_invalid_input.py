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
def test_random_projection_transformer_invalid_input():
    n_components = 'auto'
    fit_data = [[0, 1, 2]]
    for RandomProjection in all_RandomProjection:
        with pytest.raises(ValueError):
            RandomProjection(n_components=n_components).fit(fit_data)