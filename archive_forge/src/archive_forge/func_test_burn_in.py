import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_burn_in(self):
    """Tests that vectors are different after burn-in."""
    model = PoincareModel(self.data, burn_in=1, negative=3)
    original_vectors = np.copy(model.kv.vectors)
    model.train(epochs=0)
    self.assertFalse(np.allclose(model.kv.vectors, original_vectors))