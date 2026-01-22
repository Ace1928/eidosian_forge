import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi  # gamma function utils
import gensim.matutils as matutils
def test_getitem_ellipsis(self):
    assert_array_equal(self.s2c[...].sparse.toarray(), self.orig_array)