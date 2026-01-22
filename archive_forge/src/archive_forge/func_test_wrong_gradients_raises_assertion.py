import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
@unittest.skipIf(not autograd_installed, 'autograd needs to be installed for this test')
def test_wrong_gradients_raises_assertion(self):
    """Tests that discrepancy in gradients raises an error."""
    model = PoincareModel(self.data, negative=3)
    model._loss_grad = Mock(return_value=np.zeros((2 + model.negative, model.size)))
    with self.assertRaises(AssertionError):
        model.train(epochs=1, batch_size=1, check_gradients_every=1)