from unittest.mock import Mock
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sklearn.manifold import _mds as mds
from sklearn.metrics import euclidean_distances
def test_MDS():
    sim = np.array([[0, 5, 3, 4], [5, 0, 2, 2], [3, 2, 0, 1], [4, 2, 1, 0]])
    mds_clf = mds.MDS(metric=False, n_jobs=3, dissimilarity='precomputed')
    mds_clf.fit(sim)