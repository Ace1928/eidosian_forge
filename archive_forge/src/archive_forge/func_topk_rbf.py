import warnings
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import _label_propagation as label_propagation
from sklearn.utils._testing import (
def topk_rbf(X, Y=None, n_neighbors=10, gamma=1e-05):
    nn = NearestNeighbors(n_neighbors=10, metric='euclidean', n_jobs=2)
    nn.fit(X)
    W = -1 * nn.kneighbors_graph(Y, mode='distance').power(2) * gamma
    np.exp(W.data, out=W.data)
    assert issparse(W)
    return W.T