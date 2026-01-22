import itertools
import warnings
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse, stats
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.feature_selection import (
from sklearn.utils import safe_mask
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_dataframe_output_dtypes():
    """Check that the output datafarme dtypes are the same as the input.

    Non-regression test for gh-24860.
    """
    pd = pytest.importorskip('pandas')
    X, y = load_iris(return_X_y=True, as_frame=True)
    X = X.astype({'petal length (cm)': np.float32, 'petal width (cm)': np.float64})
    X['petal_width_binned'] = pd.cut(X['petal width (cm)'], bins=10)
    column_order = X.columns

    def selector(X, y):
        ranking = {'sepal length (cm)': 1, 'sepal width (cm)': 2, 'petal length (cm)': 3, 'petal width (cm)': 4, 'petal_width_binned': 5}
        return np.asarray([ranking[name] for name in column_order])
    univariate_filter = SelectKBest(selector, k=3).set_output(transform='pandas')
    output = univariate_filter.fit_transform(X, y)
    assert_array_equal(output.columns, ['petal length (cm)', 'petal width (cm)', 'petal_width_binned'])
    for name, dtype in output.dtypes.items():
        assert dtype == X.dtypes[name]