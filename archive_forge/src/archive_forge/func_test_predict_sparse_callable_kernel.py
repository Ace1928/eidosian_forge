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
def test_predict_sparse_callable_kernel(global_dtype):

    def topk_rbf(X, Y=None, n_neighbors=10, gamma=1e-05):
        nn = NearestNeighbors(n_neighbors=10, metric='euclidean', n_jobs=2)
        nn.fit(X)
        W = -1 * nn.kneighbors_graph(Y, mode='distance').power(2) * gamma
        np.exp(W.data, out=W.data)
        assert issparse(W)
        return W.T
    n_classes = 4
    n_samples = 500
    n_test = 10
    X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=20, n_informative=20, n_redundant=0, n_repeated=0, random_state=0)
    X = X.astype(global_dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=0)
    model = label_propagation.LabelSpreading(kernel=topk_rbf)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) >= 0.9
    model = label_propagation.LabelPropagation(kernel=topk_rbf)
    model.fit(X_train, y_train)
    assert model.score(X_test, y_test) >= 0.9