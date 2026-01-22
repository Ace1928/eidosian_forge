import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_dense_npfloat64(self):
    input_vector = np.random.uniform(size=(5,)).astype(np.float64)
    unit_vector = matutils.unitvec(input_vector)
    man_unit_vector = manual_unitvec(input_vector)
    self.assertTrue(np.allclose(unit_vector, man_unit_vector))
    self.assertEqual(input_vector.dtype, unit_vector.dtype)