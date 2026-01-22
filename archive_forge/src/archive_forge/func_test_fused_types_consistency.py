from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import load_iris
from sklearn.utils._seq_dataset import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('dataset_32,dataset_64', _make_fused_types_datasets())
def test_fused_types_consistency(dataset_32, dataset_64):
    NUMBER_OF_RUNS = 5
    for _ in range(NUMBER_OF_RUNS):
        (xi_data32, _, _), yi32, _, _ = dataset_32._next_py()
        (xi_data64, _, _), yi64, _, _ = dataset_64._next_py()
        assert xi_data32.dtype == np.float32
        assert xi_data64.dtype == np.float64
        assert_allclose(xi_data64, xi_data32, rtol=1e-05)
        assert_allclose(yi64, yi32, rtol=1e-05)