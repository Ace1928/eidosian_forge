import unittest
import numpy as np
from pygsp import graphs, filters
def test_simoncelli(self):
    f = filters.Simoncelli(self._G)
    self._test_methods(f, tight=True)
    f = filters.Simoncelli(self._G, a=0.25)
    self._test_methods(f, tight=True)