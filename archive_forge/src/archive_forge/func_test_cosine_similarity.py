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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_cosine_similarity(csr_container):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)
    for X_, Y_ in ((X, None), (X, Y), (Xcsr, None), (Xcsr, Ycsr)):
        K1 = pairwise_kernels(X_, Y=Y_, metric='cosine')
        X_ = normalize(X_)
        if Y_ is not None:
            Y_ = normalize(Y_)
        K2 = pairwise_kernels(X_, Y=Y_, metric='linear')
        assert_allclose(K1, K2)