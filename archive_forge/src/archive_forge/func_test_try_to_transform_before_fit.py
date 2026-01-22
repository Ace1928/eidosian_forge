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
@pytest.mark.parametrize('coo_container', COO_CONTAINERS)
def test_try_to_transform_before_fit(coo_container, global_random_seed):
    data = make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=global_random_seed, sparse_format=None)
    for RandomProjection in all_RandomProjection:
        with pytest.raises(NotFittedError):
            RandomProjection(n_components='auto').transform(data)