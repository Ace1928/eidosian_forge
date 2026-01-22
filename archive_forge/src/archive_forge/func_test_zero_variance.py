import numpy as np
import pytest
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils._testing import assert_array_equal
from sklearn.utils.fixes import BSR_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sparse_container', [None] + BSR_CONTAINERS + CSC_CONTAINERS + CSR_CONTAINERS)
def test_zero_variance(sparse_container):
    X = data if sparse_container is None else sparse_container(data)
    sel = VarianceThreshold().fit(X)
    assert_array_equal([0, 1, 3, 4], sel.get_support(indices=True))