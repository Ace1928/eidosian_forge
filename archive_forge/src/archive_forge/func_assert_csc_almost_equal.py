from io import StringIO
import tempfile
import numpy as np
from numpy.testing import assert_equal, \
from scipy.sparse import coo_matrix, csc_matrix, rand
from scipy.io import hb_read, hb_write
def assert_csc_almost_equal(r, l):
    r = csc_matrix(r)
    l = csc_matrix(l)
    assert_equal(r.indptr, l.indptr)
    assert_equal(r.indices, l.indices)
    assert_array_almost_equal_nulp(r.data, l.data, 10000)