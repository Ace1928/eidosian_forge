import logging
import os
import tempfile
import unittest
import numpy as np
from gensim.models.poincare import PoincareRelations, PoincareModel, PoincareKeyedVectors
from gensim.test.utils import datapath
def test_invalid_data_raises_error(self):
    """Tests that error is raised on invalid input data."""
    with self.assertRaises(ValueError):
        PoincareModel([('a', 'b', 'c')])
    with self.assertRaises(ValueError):
        PoincareModel(['a', 'b', 'c'])
    with self.assertRaises(ValueError):
        PoincareModel('ab')