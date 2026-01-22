import unittest
import numpy as np
import scipy.linalg
from skimage import data, img_as_float
from pygsp import graphs
def test_erdosreny(self):
    graphs.ErdosRenyi(N=100, connected=False, directed=False)
    graphs.ErdosRenyi(N=100, connected=False, directed=True)
    graphs.ErdosRenyi(N=100, connected=True, directed=False)
    graphs.ErdosRenyi(N=100, connected=True, directed=True)
    G = graphs.ErdosRenyi(N=100, p=1, self_loops=True)
    self.assertEqual(G.W.nnz, 100 ** 2)