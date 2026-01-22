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
@pytest.mark.parametrize('random_projection_cls', all_RandomProjection)
def test_random_projection_feature_names_out(coo_container, random_projection_cls, global_random_seed):
    data = make_sparse_random_data(coo_container, n_samples, n_features, n_nonzeros, random_state=global_random_seed, sparse_format=None)
    random_projection = random_projection_cls(n_components=2)
    random_projection.fit(data)
    names_out = random_projection.get_feature_names_out()
    class_name_lower = random_projection_cls.__name__.lower()
    expected_names_out = np.array([f'{class_name_lower}{i}' for i in range(random_projection.n_components_)], dtype=object)
    assert_array_equal(names_out, expected_names_out)