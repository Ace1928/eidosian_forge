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
def test_no_data_conversion_warning():
    rng = np.random.RandomState(0)
    X = rng.randn(5, 4)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DataConversionWarning)
        pairwise_distances(X, metric='minkowski')