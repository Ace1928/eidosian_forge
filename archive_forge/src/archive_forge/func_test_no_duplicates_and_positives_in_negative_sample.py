import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_no_duplicates_and_positives_in_negative_sample(self):
    """Tests that no duplicates or positively related nodes are present in negative samples."""
    model = PoincareModel(self.data_large, negative=3)
    positive_nodes = model.node_relations[0]
    num_samples = 100
    for i in range(num_samples):
        negatives = model._sample_negatives(0)
        self.assertFalse(positive_nodes & set(negatives))
        self.assertEqual(len(negatives), len(set(negatives)))