import numpy as np
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('keep_empty_features', [True, False])
@pytest.mark.parametrize('imputer', imputers(), ids=lambda x: x.__class__.__name__)
def test_keep_empty_features(imputer, keep_empty_features):
    """Check that the imputer keeps features with only missing values."""
    X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])
    imputer = imputer.set_params(add_indicator=False, keep_empty_features=keep_empty_features)
    for method in ['fit_transform', 'transform']:
        X_imputed = getattr(imputer, method)(X)
        if keep_empty_features:
            assert X_imputed.shape == X.shape
        else:
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)