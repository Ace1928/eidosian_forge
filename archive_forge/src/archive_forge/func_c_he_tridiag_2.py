from ..libmp.backend import xrange
from .eigen import defun
def c_he_tridiag_2(ctx, A, T, B):
    """
    This routine applied the unitary matrix Q described in c_he_tridiag_0
    onto the the matrix B, i.e. it forms Q*B.

    parameters:
      A    (input) On input, A is the same matrix as delivered by c_he_tridiag_0.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

      B    (input/output) On input, B is a complex matrix. On output B is replaced
           by Q*B.

    This routine is a python translation of the fortran routine htribk.f in the
    software library EISPACK (see netlib.org). See c_he_tridiag_0 for more
    references.
    """
    n = A.rows
    for i in xrange(0, n):
        for k in xrange(0, n):
            B[k, i] *= T[k]
    for i in xrange(0, n):
        if A[i, i] != 0:
            for j in xrange(0, n):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i, k]) * B[k, j]
                for k in xrange(0, i):
                    B[k, j] -= G * A[k, i]