import numpy as np
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.filterwarnings('ignore::sklearn.exceptions.ConvergenceWarning')
@pytest.mark.parametrize('imputer', imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize('add_indicator', [True, False])
def test_imputers_pandas_na_integer_array_support(imputer, add_indicator):
    pd = pytest.importorskip('pandas')
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)
    X = np.array([[marker, 1, 5, marker, 1], [2, marker, 1, marker, 2], [6, 3, marker, marker, 3], [1, 2, 9, marker, 4]])
    X_trans_expected = imputer.fit_transform(X)
    X_df = pd.DataFrame(X, dtype='Int16', columns=['a', 'b', 'c', 'd', 'e'])
    X_trans = imputer.fit_transform(X_df)
    assert_allclose(X_trans_expected, X_trans)