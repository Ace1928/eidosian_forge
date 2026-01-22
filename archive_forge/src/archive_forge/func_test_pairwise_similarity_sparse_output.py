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
@pytest.mark.parametrize('metric, pairwise_func', [('linear', linear_kernel), ('cosine', cosine_similarity)])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_pairwise_similarity_sparse_output(metric, pairwise_func, csr_container):
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((3, 4))
    Xcsr = csr_container(X)
    Ycsr = csr_container(Y)
    K1 = pairwise_func(Xcsr, Ycsr, dense_output=False)
    assert issparse(K1)
    K2 = pairwise_func(X, Y, dense_output=True)
    assert not issparse(K2)
    assert_allclose(K1.toarray(), K2)
    K3 = pairwise_kernels(X, Y=Y, metric=metric)
    assert_allclose(K1.toarray(), K3)