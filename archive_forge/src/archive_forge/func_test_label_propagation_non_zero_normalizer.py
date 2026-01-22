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
@pytest.mark.parametrize('LabelPropagationCls', [label_propagation.LabelSpreading, label_propagation.LabelPropagation])
def test_label_propagation_non_zero_normalizer(LabelPropagationCls):
    X = np.array([[100.0, 100.0], [100.0, 100.0], [0.0, 0.0], [0.0, 0.0]])
    y = np.array([0, 1, -1, -1])
    mdl = LabelPropagationCls(kernel='knn', max_iter=100, n_neighbors=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        mdl.fit(X, y)