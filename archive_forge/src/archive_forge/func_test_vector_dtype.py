import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_vector_dtype(self):
    """Tests whether vectors have the correct dtype before and after training."""
    model = PoincareModel(self.data_large, dtype=np.float32, burn_in=0, negative=3)
    self.assertEqual(model.kv.vectors.dtype, np.float32)
    model.train(epochs=1)
    self.assertEqual(model.kv.vectors.dtype, np.float32)