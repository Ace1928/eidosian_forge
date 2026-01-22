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
@pytest.mark.parametrize('good_reduce', [lambda D, start: list(D), lambda D, start: np.array(D), lambda D, start: (list(D), list(D))] + [lambda D, start, scipy_csr_type=scipy_csr_type: scipy_csr_type(D) for scipy_csr_type in CSR_CONTAINERS] + [lambda D, start, scipy_dok_type=scipy_dok_type: (scipy_dok_type(D), np.array(D), list(D)) for scipy_dok_type in DOK_CONTAINERS])
def test_pairwise_distances_chunked_reduce_valid(good_reduce):
    X = np.arange(10).reshape(-1, 1)
    S_chunks = pairwise_distances_chunked(X, None, reduce_func=good_reduce, working_memory=64)
    next(S_chunks)