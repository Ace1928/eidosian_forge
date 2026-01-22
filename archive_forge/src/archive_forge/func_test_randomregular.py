import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_randomregular(self):
    k = 6
    G = graphs.RandomRegular(k=k)
    np.testing.assert_equal(G.W.sum(0), k)
    np.testing.assert_equal(G.W.sum(1), k)