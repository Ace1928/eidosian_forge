import unittest
import numpy as np
from pygsp import graphs, filters
def test_custom_filter(self):

    def kernel(x):
        return x / (1.0 + x)
    f = filters.Filter(self._G, kernels=kernel)
    self.assertEqual(f.Nf, 1)
    self.assertIs(f._kernels[0], kernel)
    self._test_methods(f, tight=False)