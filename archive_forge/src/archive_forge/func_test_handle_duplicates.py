import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_handle_duplicates(self):
    """Tests that correct number of negatives are used."""
    vector_updates = np.array([[0.5, 0.5], [0.1, 0.2], [0.3, -0.2]])
    node_indices = [0, 1, 0]
    PoincareModel._handle_duplicates(vector_updates, node_indices)
    vector_updates_expected = np.array([[0.0, 0.0], [0.1, 0.2], [0.8, 0.3]])
    self.assertTrue((vector_updates == vector_updates_expected).all())