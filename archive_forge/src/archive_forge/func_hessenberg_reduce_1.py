from ..libmp.backend import xrange
def hessenberg_reduce_1(ctx, A, T):
    """
    This routine forms the unitary matrix Q described in hessenberg_reduce_0.

    parameters:
      A    (input/output) On input, A is the same matrix as delivered by
           hessenberg_reduce_0. On output, A is set to Q.

      T    (input) On input, T is the same array as delivered by hessenberg_reduce_0.
    """
    n = A.rows
    if n == 1:
        A[0, 0] = 1
        return
    A[0, 0] = A[1, 1] = 1
    A[0, 1] = A[1, 0] = 0
    for i in xrange(2, n):
        if T[i] != 0:
            for j in xrange(0, i):
                G = T[i] * A[i - 1, j]
                for k in xrange(0, i - 1):
                    G += A[i, k] * A[k, j]
                A[i - 1, j] -= G * ctx.conj(T[i])
                for k in xrange(0, i - 1):
                    A[k, j] -= G * ctx.conj(A[i, k])
        A[i, i] = 1
        for j in xrange(0, i):
            A[j, i] = A[i, j] = 0