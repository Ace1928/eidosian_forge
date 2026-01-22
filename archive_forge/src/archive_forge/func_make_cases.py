from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
def make_cases(original, dtype):
    cases = []
    cases.append((matrix(original, dtype=dtype), original))
    cases.append((np.array(original, dtype=dtype), original))
    cases.append((sparse.csr_matrix(original, dtype=dtype), original))

    def mv(x, dtype):
        y = original.dot(x)
        if len(x.shape) == 2:
            y = y.reshape(-1, 1)
        return y

    def rmv(x, dtype):
        return original.T.conj().dot(x)

    class BaseMatlike(interface.LinearOperator):
        args = ()

        def __init__(self, dtype):
            self.dtype = np.dtype(dtype)
            self.shape = original.shape

        def _matvec(self, x):
            return mv(x, self.dtype)

    class HasRmatvec(BaseMatlike):
        args = ()

        def _rmatvec(self, x):
            return rmv(x, self.dtype)

    class HasAdjoint(BaseMatlike):
        args = ()

        def _adjoint(self):
            shape = (self.shape[1], self.shape[0])
            matvec = partial(rmv, dtype=self.dtype)
            rmatvec = partial(mv, dtype=self.dtype)
            return interface.LinearOperator(matvec=matvec, rmatvec=rmatvec, dtype=self.dtype, shape=shape)

    class HasRmatmat(HasRmatvec):

        def _matmat(self, x):
            return original.dot(x)

        def _rmatmat(self, x):
            return original.T.conj().dot(x)
    cases.append((HasRmatvec(dtype), original))
    cases.append((HasAdjoint(dtype), original))
    cases.append((HasRmatmat(dtype), original))
    return cases