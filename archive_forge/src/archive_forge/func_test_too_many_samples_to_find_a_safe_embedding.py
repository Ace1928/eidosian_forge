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
def test_too_many_samples_to_find_a_safe_embedding(coo_container, global_random_seed):
    data = make_sparse_random_data(coo_container, n_samples=1000, n_features=100, n_nonzeros=1000, random_state=global_random_seed, sparse_format=None)
    for RandomProjection in all_RandomProjection:
        rp = RandomProjection(n_components='auto', eps=0.1)
        expected_msg = 'eps=0.100000 and n_samples=1000 lead to a target dimension of 5920 which is larger than the original space with n_features=100'
        with pytest.raises(ValueError, match=expected_msg):
            rp.fit(data)