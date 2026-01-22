import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('metric, func', PAIRED_DISTANCES.items())
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_paired_distances(metric, func, csr_container):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((5, 4))
    S = paired_distances(X, Y, metric=metric)
    S2 = func(X, Y)
    assert_allclose(S, S2)
    S3 = func(csr_container(X), csr_container(Y))
    assert_allclose(S, S3)
    if metric in PAIRWISE_DISTANCE_FUNCTIONS:
        distances = PAIRWISE_DISTANCE_FUNCTIONS[metric](X, Y)
        distances = np.diag(distances)
        assert_allclose(distances, S)