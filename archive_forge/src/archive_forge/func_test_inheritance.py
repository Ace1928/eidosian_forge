from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
def test_inheritance():

    class Empty(interface.LinearOperator):
        pass
    with warns(RuntimeWarning, match='should implement at least'):
        assert_raises(TypeError, Empty)

    class Identity(interface.LinearOperator):

        def __init__(self, n):
            super().__init__(dtype=None, shape=(n, n))

        def _matvec(self, x):
            return x
    id3 = Identity(3)
    assert_equal(id3.matvec([1, 2, 3]), [1, 2, 3])
    assert_raises(NotImplementedError, id3.rmatvec, [4, 5, 6])

    class MatmatOnly(interface.LinearOperator):

        def __init__(self, A):
            super().__init__(A.dtype, A.shape)
            self.A = A

        def _matmat(self, x):
            return self.A.dot(x)
    mm = MatmatOnly(np.random.randn(5, 3))
    assert_equal(mm.matvec(np.random.randn(3)).shape, (5,))