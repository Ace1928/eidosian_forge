import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_error_if_negative_more_than_population(self):
    """Tests error is rased if number of negatives to sample is more than remaining nodes."""
    model = PoincareModel(self.data, negative=5)
    with self.assertRaises(ValueError):
        model.train(epochs=1)