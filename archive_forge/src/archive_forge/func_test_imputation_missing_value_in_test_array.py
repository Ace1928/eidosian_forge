import numpy as np
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('imputer', imputers(), ids=lambda x: x.__class__.__name__)
def test_imputation_missing_value_in_test_array(imputer):
    train = [[1], [2]]
    test = [[3], [np.nan]]
    imputer.set_params(add_indicator=True)
    imputer.fit(train).transform(test)