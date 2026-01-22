import unittest
import numpy as np
from pygsp import graphs, filters
def test_heat(self):
    f = filters.Heat(self._G, normalize=False, tau=10)
    self._test_methods(f, tight=False)
    f = filters.Heat(self._G, normalize=False, tau=np.array([5, 10]))
    self._test_methods(f, tight=False)
    f = filters.Heat(self._G, normalize=True, tau=10)
    np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)), 1)
    self._test_methods(f, tight=False)
    f = filters.Heat(self._G, normalize=True, tau=[5, 10])
    np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)[0]), 1)
    np.testing.assert_allclose(np.linalg.norm(f.evaluate(self._G.e)[1]), 1)
    self._test_methods(f, tight=False)