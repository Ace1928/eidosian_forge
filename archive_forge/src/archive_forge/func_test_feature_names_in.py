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
def test_feature_names_in():
    """Check that feature_name_in are recorded by `_validate_data`"""
    pd = pytest.importorskip('pandas')
    iris = datasets.load_iris()
    X_np = iris.data
    df = pd.DataFrame(X_np, columns=iris.feature_names)

    class NoOpTransformer(TransformerMixin, BaseEstimator):

        def fit(self, X, y=None):
            self._validate_data(X)
            return self

        def transform(self, X):
            self._validate_data(X, reset=False)
            return X
    trans = NoOpTransformer().fit(df)
    assert_array_equal(trans.feature_names_in_, df.columns)
    trans.fit(X_np)
    assert not hasattr(trans, 'feature_names_in_')
    trans.fit(df)
    msg = 'The feature names should match those that were passed'
    df_bad = pd.DataFrame(X_np, columns=iris.feature_names[::-1])
    with pytest.raises(ValueError, match=msg):
        trans.transform(df_bad)
    msg = 'X does not have valid feature names, but NoOpTransformer was fitted with feature names'
    with pytest.warns(UserWarning, match=msg):
        trans.transform(X_np)
    msg = 'X has feature names, but NoOpTransformer was fitted without feature names'
    trans = NoOpTransformer().fit(X_np)
    with pytest.warns(UserWarning, match=msg):
        trans.transform(df)
    df_int_names = pd.DataFrame(X_np)
    trans = NoOpTransformer()
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        trans.fit(df_int_names)
    Xs = [X_np, df_int_names]
    for X in Xs:
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            trans.transform(X)
    df_mixed = pd.DataFrame(X_np, columns=['a', 'b', 1, 2])
    trans = NoOpTransformer()
    msg = re.escape("Feature names are only supported if all input features have string names, but your input has ['int', 'str'] as feature name / column name types. If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.")
    with pytest.raises(TypeError, match=msg):
        trans.fit(df_mixed)
    with pytest.raises(TypeError, match=msg):
        trans.transform(df_mixed)