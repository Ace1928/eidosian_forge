import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, scale
from sklearn.utils import parallel_backend
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('n_jobs', [1, 2])
@pytest.mark.parametrize('max_samples', [0.5, 1.0])
def test_permutation_importance_correlated_feature_regression_pandas(n_jobs, max_samples):
    pd = pytest.importorskip('pandas')
    rng = np.random.RandomState(42)
    n_repeats = 5
    dataset = load_iris()
    X, y = (dataset.data, dataset.target)
    y_with_little_noise = (y + rng.normal(scale=0.001, size=y.shape[0])).reshape(-1, 1)
    X = pd.DataFrame(X, columns=dataset.feature_names)
    X['correlated_feature'] = y_with_little_noise
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    result = permutation_importance(clf, X, y, n_repeats=n_repeats, random_state=rng, n_jobs=n_jobs, max_samples=max_samples)
    assert result.importances.shape == (X.shape[1], n_repeats)
    assert np.all(result.importances_mean[-1] > result.importances_mean[:-1])