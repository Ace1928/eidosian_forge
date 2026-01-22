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
def test_KNeighborsClassifier_raise_on_all_zero_weights():
    """Check that `predict` and `predict_proba` raises on sample of all zeros weights.

    Related to Issue #25854.
    """
    X = [[0, 1], [1, 2], [2, 3], [3, 4]]
    y = [0, 0, 1, 1]

    def _weights(dist):
        return np.vectorize(lambda x: 0 if x > 0.5 else 1)(dist)
    est = neighbors.KNeighborsClassifier(n_neighbors=3, weights=_weights)
    est.fit(X, y)
    msg = "All neighbors of some sample is getting zero weights. Please modify 'weights' to avoid this case if you are using a user-defined function."
    with pytest.raises(ValueError, match=msg):
        est.predict([[1.1, 1.1]])
    with pytest.raises(ValueError, match=msg):
        est.predict_proba([[1.1, 1.1]])