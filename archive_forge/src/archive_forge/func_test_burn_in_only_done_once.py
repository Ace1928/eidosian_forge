import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_burn_in_only_done_once(self):
    """Tests that burn-in does not happen when train is called a second time."""
    model = PoincareModel(self.data, negative=3, burn_in=1)
    model.train(epochs=0)
    original_vectors = np.copy(model.kv.vectors)
    model.train(epochs=0)
    self.assertTrue(np.allclose(model.kv.vectors, original_vectors))