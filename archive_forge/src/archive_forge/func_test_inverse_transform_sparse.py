import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_inverse_transform_sparse(csc_container):
    X_sp = csc_container(X)
    Xt_sp = csc_container(Xt)
    sel = StepSelector()
    Xinv_actual = sel.fit(X_sp).inverse_transform(Xt_sp)
    assert_array_equal(Xinv, Xinv_actual.toarray())
    assert np.int32 == sel.inverse_transform(Xt_sp.astype(np.int32)).dtype
    assert np.float32 == sel.inverse_transform(Xt_sp.astype(np.float32)).dtype
    with pytest.raises(ValueError):
        sel.inverse_transform(np.array([[1], [2]]))