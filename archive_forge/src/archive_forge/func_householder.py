from copy import copy
from ..libmp.backend import xrange
def householder(ctx, A):
    """
        (A|b) -> H, p, x, res

        (A|b) is the coefficient matrix with left hand side of an optionally
        overdetermined linear equation system.
        H and p contain all information about the transformation matrices.
        x is the solution, res the residual.
        """
    if not isinstance(A, ctx.matrix):
        raise TypeError('A should be a type of ctx.matrix')
    m = A.rows
    n = A.cols
    if m < n - 1:
        raise RuntimeError('Columns should not be less than rows')
    p = []
    for j in xrange(0, n - 1):
        s = ctx.fsum((abs(A[i, j]) ** 2 for i in xrange(j, m)))
        if not abs(s) > ctx.eps:
            raise ValueError('matrix is numerically singular')
        p.append(-ctx.sign(ctx.re(A[j, j])) * ctx.sqrt(s))
        kappa = ctx.one / (s - p[j] * A[j, j])
        A[j, j] -= p[j]
        for k in xrange(j + 1, n):
            y = ctx.fsum((ctx.conj(A[i, j]) * A[i, k] for i in xrange(j, m))) * kappa
            for i in xrange(j, m):
                A[i, k] -= A[i, j] * y
    x = [A[i, n - 1] for i in xrange(n - 1)]
    for i in xrange(n - 2, -1, -1):
        x[i] -= ctx.fsum((A[i, j] * x[j] for j in xrange(i + 1, n - 1)))
        x[i] /= p[i]
    if not m == n - 1:
        r = [A[m - 1 - i, n - 1] for i in xrange(m - n + 1)]
    else:
        r = [0] * m
    return (A, p, x, r)