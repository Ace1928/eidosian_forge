import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_vector_shape(self):
    """Tests whether vectors are initialized with the correct size."""
    model = PoincareModel(self.data, size=20)
    self.assertEqual(model.kv.vectors.shape, (7, 20))