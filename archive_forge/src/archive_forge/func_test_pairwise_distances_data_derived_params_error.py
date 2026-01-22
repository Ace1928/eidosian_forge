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
@pytest.mark.parametrize('metric', ['seuclidean', 'mahalanobis'])
def test_pairwise_distances_data_derived_params_error(metric):
    rng = np.random.RandomState(0)
    X = rng.random_sample((100, 10))
    Y = rng.random_sample((100, 10))
    with pytest.raises(ValueError, match=f"The '(V|VI)' parameter is required for the {metric} metric"):
        pairwise_distances(X, Y, metric=metric)