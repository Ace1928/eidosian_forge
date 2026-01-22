import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_differential_operator(self):
    G = graphs.StochasticBlockModel(N=100, directed=False)
    L = G.D.T.dot(G.D)
    np.testing.assert_allclose(L.toarray(), G.L.toarray())
    G = graphs.StochasticBlockModel(N=100, directed=True)
    self.assertRaises(NotImplementedError, G.compute_differential_operator)