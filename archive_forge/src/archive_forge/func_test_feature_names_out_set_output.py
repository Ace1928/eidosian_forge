import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
@pytest.mark.parametrize('y, feature_names', [([1, 2] * 10, ['A', 'B']), ([1, 2, 3] * 6 + [1, 2], ['A_1', 'A_2', 'A_3', 'B_1', 'B_2', 'B_3']), (['y1', 'y2', 'y3'] * 6 + ['y1', 'y2'], ['A_y1', 'A_y2', 'A_y3', 'B_y1', 'B_y2', 'B_y3'])])
def test_feature_names_out_set_output(y, feature_names):
    """Check TargetEncoder works with set_output."""
    pd = pytest.importorskip('pandas')
    X_df = pd.DataFrame({'A': ['a', 'b'] * 10, 'B': [1, 2] * 10})
    enc_default = TargetEncoder(cv=2, smooth=3.0, random_state=0)
    enc_default.set_output(transform='default')
    enc_pandas = TargetEncoder(cv=2, smooth=3.0, random_state=0)
    enc_pandas.set_output(transform='pandas')
    X_default = enc_default.fit_transform(X_df, y)
    X_pandas = enc_pandas.fit_transform(X_df, y)
    assert_allclose(X_pandas.to_numpy(), X_default)
    assert_array_equal(enc_pandas.get_feature_names_out(), feature_names)
    assert_array_equal(enc_pandas.get_feature_names_out(), X_pandas.columns)