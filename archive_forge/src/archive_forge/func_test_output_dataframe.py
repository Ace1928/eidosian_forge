import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.fixes import CSC_CONTAINERS
def test_output_dataframe():
    """Check output dtypes for dataframes is consistent with the input dtypes."""
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'a': pd.Series([1.0, 2.4, 4.5], dtype=np.float32), 'b': pd.Series(['a', 'b', 'a'], dtype='category'), 'c': pd.Series(['j', 'b', 'b'], dtype='category'), 'd': pd.Series([3.0, 2.4, 1.2], dtype=np.float64)})
    for step in [2, 3]:
        sel = StepSelector(step=step).set_output(transform='pandas')
        sel.fit(X)
        output = sel.transform(X)
        for name, dtype in output.dtypes.items():
            assert dtype == X.dtypes[name]
    sel0 = StepSelector(step=0).set_output(transform='pandas')
    sel0.fit(X, y)
    msg = 'No features were selected'
    with pytest.warns(UserWarning, match=msg):
        output0 = sel0.transform(X)
    assert_array_equal(output0.index, X.index)
    assert output0.shape == (X.shape[0], 0)