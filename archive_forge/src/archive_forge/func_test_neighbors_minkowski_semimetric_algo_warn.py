import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('Estimator', [neighbors.KNeighborsClassifier, neighbors.RadiusNeighborsClassifier, neighbors.KNeighborsRegressor, neighbors.RadiusNeighborsRegressor])
@pytest.mark.parametrize('n_features', [2, 100])
@pytest.mark.parametrize('algorithm', ['auto', 'brute'])
def test_neighbors_minkowski_semimetric_algo_warn(Estimator, n_features, algorithm):
    """
    Validation of all classes extending NeighborsBase with
    Minkowski semi-metrics (i.e. when 0 < p < 1). That proper
    Warning is raised for `algorithm="auto"` and "brute".
    """
    X = rng.random_sample((10, n_features))
    y = np.ones(10)
    model = Estimator(p=0.1, algorithm=algorithm)
    msg = "Mind that for 0 < p < 1, Minkowski metrics are not distance metrics. Continuing the execution with `algorithm='brute'`."
    with pytest.warns(UserWarning, match=msg):
        model.fit(X, y)
    assert model._fit_method == 'brute'