import logging
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from gensim.corpora import mmcorpus
from gensim.models import normmodel
from gensim.test.utils import datapath, get_tmpfile
def test_numpyndarrayInput_l2(self):
    """Test for np ndarray input for l2 transformation"""
    ndarray_matrix = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
    normalized = self.model_l2.normalize(ndarray_matrix)
    self.assertTrue(isinstance(normalized, np.ndarray))
    expected = np.array([[0.10482848, 0.0, 0.20965697], [0.0, 0.0, 0.31448545], [0.41931393, 0.52414242, 0.6289709]])
    self.assertTrue(np.allclose(normalized, expected))
    self.assertRaises(ValueError, lambda model, doc: model.normalize(doc), self.model_l2, [1, 2, 3])