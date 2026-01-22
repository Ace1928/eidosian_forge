import warnings
import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import (
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tests.test_tree import assert_is_subtree
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('features, expected_pd_shape', [(0, (3, 10)), (iris.feature_names[0], (3, 10)), ([0, 2], (3, 10, 10)), ([iris.feature_names[i] for i in (0, 2)], (3, 10, 10)), ([True, False, True, False], (3, 10, 10))], ids=['scalar-int', 'scalar-str', 'list-int', 'list-str', 'mask'])
def test_partial_dependence_feature_type(features, expected_pd_shape):
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    preprocessor = make_column_transformer((StandardScaler(), [iris.feature_names[i] for i in (0, 2)]), (RobustScaler(), [iris.feature_names[i] for i in (1, 3)]))
    pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=0))
    pipe.fit(df, iris.target)
    pdp_pipe = partial_dependence(pipe, df, features=features, grid_resolution=10, kind='average')
    assert pdp_pipe['average'].shape == expected_pd_shape
    assert len(pdp_pipe['grid_values']) == len(pdp_pipe['average'].shape) - 1