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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_predict_sparse_ball_kd_tree(csr_container):
    rng = np.random.RandomState(0)
    X = rng.rand(5, 5)
    y = rng.randint(0, 2, 5)
    nbrs1 = neighbors.KNeighborsClassifier(1, algorithm='kd_tree')
    nbrs2 = neighbors.KNeighborsRegressor(1, algorithm='ball_tree')
    for model in [nbrs1, nbrs2]:
        model.fit(X, y)
        with pytest.raises(ValueError):
            model.predict(csr_container(X))