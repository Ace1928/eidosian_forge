import time
import joblib
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import config_context, get_config
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('n_jobs', [1, 2])
def test_dispatch_config_parallel(n_jobs):
    """Check that we properly dispatch the configuration in parallel processing.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/25239
    """
    pd = pytest.importorskip('pandas')
    iris = load_iris(as_frame=True)

    class TransformerRequiredDataFrame(StandardScaler):

        def fit(self, X, y=None):
            assert isinstance(X, pd.DataFrame), 'X should be a DataFrame'
            return super().fit(X, y)

        def transform(self, X, y=None):
            assert isinstance(X, pd.DataFrame), 'X should be a DataFrame'
            return super().transform(X, y)
    dropper = make_column_transformer(('drop', [0]), remainder='passthrough', n_jobs=n_jobs)
    param_grid = {'randomforestclassifier__max_depth': [1, 2, 3]}
    search_cv = GridSearchCV(make_pipeline(dropper, TransformerRequiredDataFrame(), RandomForestClassifier(n_estimators=5, n_jobs=n_jobs)), param_grid, cv=5, n_jobs=n_jobs, error_score='raise')
    with pytest.raises(AssertionError, match='X should be a DataFrame'):
        search_cv.fit(iris.data, iris.target)
    with config_context(transform_output='pandas'):
        search_cv.fit(iris.data, iris.target)
    assert not np.isnan(search_cv.cv_results_['mean_test_score']).any()