from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
def test_no_double_init():
    call_count = [0]

    def matvec(v):
        call_count[0] += 1
        return v
    interface.LinearOperator((2, 2), matvec=matvec)
    assert_equal(call_count[0], 1)