import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
def test_validate_data_cast_to_ndarray():
    """Check cast_to_ndarray option of _validate_data."""
    pd = pytest.importorskip('pandas')
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    class NoOpTransformer(TransformerMixin, BaseEstimator):
        pass
    no_op = NoOpTransformer()
    X_np_out = no_op._validate_data(df, cast_to_ndarray=True)
    assert isinstance(X_np_out, np.ndarray)
    assert_allclose(X_np_out, df.to_numpy())
    X_df_out = no_op._validate_data(df, cast_to_ndarray=False)
    assert X_df_out is df
    y_np_out = no_op._validate_data(y=y, cast_to_ndarray=True)
    assert isinstance(y_np_out, np.ndarray)
    assert_allclose(y_np_out, y.to_numpy())
    y_series_out = no_op._validate_data(y=y, cast_to_ndarray=False)
    assert y_series_out is y
    X_np_out, y_np_out = no_op._validate_data(df, y, cast_to_ndarray=True)
    assert isinstance(X_np_out, np.ndarray)
    assert_allclose(X_np_out, df.to_numpy())
    assert isinstance(y_np_out, np.ndarray)
    assert_allclose(y_np_out, y.to_numpy())
    X_df_out, y_series_out = no_op._validate_data(df, y, cast_to_ndarray=False)
    assert X_df_out is df
    assert y_series_out is y
    msg = 'Validation should be done on X, y or both.'
    with pytest.raises(ValueError, match=msg):
        no_op._validate_data()