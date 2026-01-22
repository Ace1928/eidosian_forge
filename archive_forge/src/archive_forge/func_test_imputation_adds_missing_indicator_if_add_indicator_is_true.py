import numpy as np
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('imputer', imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize('missing_value_test', [np.nan, 1])
def test_imputation_adds_missing_indicator_if_add_indicator_is_true(imputer, missing_value_test):
    """Check that missing indicator always exists when add_indicator=True.

    Non-regression test for gh-26590.
    """
    X_train = np.array([[0, np.nan], [1, 2]])
    X_test = np.array([[0, missing_value_test], [1, 2]])
    imputer.set_params(add_indicator=True)
    imputer.fit(X_train)
    X_test_imputed_with_indicator = imputer.transform(X_test)
    assert X_test_imputed_with_indicator.shape == (2, 3)
    imputer.set_params(add_indicator=False)
    imputer.fit(X_train)
    X_test_imputed_without_indicator = imputer.transform(X_test)
    assert X_test_imputed_without_indicator.shape == (2, 2)
    assert_allclose(X_test_imputed_with_indicator[:, :-1], X_test_imputed_without_indicator)
    if np.isnan(missing_value_test):
        expected_missing_indicator = [1, 0]
    else:
        expected_missing_indicator = [0, 0]
    assert_allclose(X_test_imputed_with_indicator[:, -1], expected_missing_indicator)