import logging
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from gensim.corpora import mmcorpus
from gensim.models import normmodel
from gensim.test.utils import datapath, get_tmpfile
def test_sparseCSRInput_l1(self):
    """Test sparse csr matrix input for l1 transformation"""
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    sparse_matrix = csr_matrix((data, (row, col)), shape=(3, 3))
    normalized = self.model_l1.normalize(sparse_matrix)
    self.assertTrue(issparse(normalized))
    expected = np.array([[0.04761905, 0.0, 0.0952381], [0.0, 0.0, 0.14285714], [0.19047619, 0.23809524, 0.28571429]])
    self.assertTrue(np.allclose(normalized.toarray(), expected))