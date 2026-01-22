import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('csc_container', CSC_CONTAINERS)
def test_transform_sparse(csc_container):
    X_sp = csc_container(X)
    sel = StepSelector()
    Xt_actual = sel.fit(X_sp).transform(X_sp)
    Xt_actual2 = sel.fit_transform(X_sp)
    assert_array_equal(Xt, Xt_actual.toarray())
    assert_array_equal(Xt, Xt_actual2.toarray())
    assert np.int32 == sel.transform(X_sp.astype(np.int32)).dtype
    assert np.float32 == sel.transform(X_sp.astype(np.float32)).dtype
    with pytest.raises(ValueError):
        sel.transform(np.array([[1], [2]]))