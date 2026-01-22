import logging
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from gensim.corpora import mmcorpus
from gensim.models import normmodel
from gensim.test.utils import datapath, get_tmpfile
def test_sparseCSRInput_l2(self):
    """Test sparse csr matrix input for l2 transformation"""
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))
    normalized = self.model_l2.normalize(sparse_matrix)
    self.assertTrue(issparse(normalized))
    expected = np.array([[0.10482848, 0.0, 0.20965697], [0.0, 0.0, 0.31448545], [0.41931393, 0.52414242, 0.6289709]])
    self.assertTrue(np.allclose(normalized.toarray(), expected))