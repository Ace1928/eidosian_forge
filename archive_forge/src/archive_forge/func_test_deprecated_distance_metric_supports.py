import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('metric', sorted(list(NearestCentroid._valid_metrics - {'manhattan', 'euclidean'})))
def test_deprecated_distance_metric_supports(metric):
    clf = NearestCentroid(metric=metric)
    with pytest.warns(FutureWarning, match='Support for distance metrics other than euclidean and manhattan'):
        clf.fit(X, y)