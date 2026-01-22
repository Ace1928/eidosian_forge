import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_difference_in_hierarchy(self):
    """Test difference_in_hierarchy returns expected value for two nodes, and for identical nodes."""
    self.assertTrue(np.allclose(self.vectors.difference_in_hierarchy('dog.n.01', 'dog.n.01'), 0))
    self.assertTrue(np.allclose(self.vectors.difference_in_hierarchy('mammal.n.01', 'dog.n.01'), 0.9384287))
    self.assertTrue(np.allclose(self.vectors.difference_in_hierarchy('dog.n.01', 'mammal.n.01'), -0.9384287))