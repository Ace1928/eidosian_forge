import unittest
import numpy as np
from pygsp import graphs, filters
def test_mexicanhat(self):
    f = filters.MexicanHat(self._G, Nf=5, normalize=False)
    self._test_methods(f, tight=False)
    f = filters.MexicanHat(self._G, Nf=4, normalize=True)
    self._test_methods(f, tight=False)