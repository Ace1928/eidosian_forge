import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_reproducible(self):
    """Tests that vectors are same for two independent models trained with the same seed."""
    model_1 = PoincareModel(self.data_large, seed=1, negative=3, burn_in=1)
    model_1.train(epochs=2)
    model_2 = PoincareModel(self.data_large, seed=1, negative=3, burn_in=1)
    model_2.train(epochs=2)
    self.assertTrue(np.allclose(model_1.kv.vectors, model_2.kv.vectors))