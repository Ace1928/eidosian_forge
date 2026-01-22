import warnings
import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation, affinity_propagation
from sklearn.cluster._affinity_propagation import _equal_similarities_and_preferences
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import euclidean_distances
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_affinity_propagation_precomputed_with_sparse_input(csr_container):
    err_msg = 'Sparse data was passed for X, but dense data is required'
    with pytest.raises(TypeError, match=err_msg):
        AffinityPropagation(affinity='precomputed').fit(csr_container((3, 3)))