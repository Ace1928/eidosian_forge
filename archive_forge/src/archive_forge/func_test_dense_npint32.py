import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_dense_npint32(self):
    input_vector = np.random.randint(10, size=5).astype(np.int32)
    unit_vector = matutils.unitvec(input_vector)
    man_unit_vector = manual_unitvec(input_vector)
    self.assertTrue(np.allclose(unit_vector, man_unit_vector))
    self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))