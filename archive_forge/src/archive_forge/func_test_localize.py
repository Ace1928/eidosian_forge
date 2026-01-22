import unittest
import numpy as np
from pygsp import graphs, filters
def test_localize(self):
    G = graphs.Grid2d(20)
    G.compute_fourier_basis()
    g = filters.Heat(G, 100)
    NODE = 10
    s1 = g.localize(NODE, method='exact')
    gL = G.U.dot(np.diag(g.evaluate(G.e)[0]).dot(G.U.T))
    s2 = np.sqrt(G.N) * gL[NODE, :]
    np.testing.assert_allclose(s1, s2)
    F = g.compute_frame(method='exact')
    np.testing.assert_allclose(F, gL)