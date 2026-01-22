from ..libmp.backend import xrange
def hessenberg_qr(ctx, A, Q):
    """
    This routine computes the Schur decomposition of an upper Hessenberg matrix A.
    Given A, an unitary matrix Q is determined such that

          Q' A Q = R                   and                  Q' Q = Q Q' = 1

    where R is an upper right triangular matrix. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation).

    parameters:
      A         (input/output) On input, A contains an upper Hessenberg matrix.
                On output, A is replace by the upper right triangluar matrix R.

      Q         (input/output) The parameter Q is multiplied by the unitary
                matrix Q arising from the Schur decomposition. Q can also be
                false, in which case the unitary matrix Q is not computated.
    """
    n = A.rows
    norm = 0
    for x in xrange(n):
        for y in xrange(min(x + 2, n)):
            norm += ctx.re(A[y, x]) ** 2 + ctx.im(A[y, x]) ** 2
    norm = ctx.sqrt(norm) / n
    if norm == 0:
        return
    n0 = 0
    n1 = n
    eps = ctx.eps / (100 * n)
    maxits = ctx.dps * 4
    its = totalits = 0
    while 1:
        k = n0
        while k + 1 < n1:
            s = abs(ctx.re(A[k, k])) + abs(ctx.im(A[k, k])) + abs(ctx.re(A[k + 1, k + 1])) + abs(ctx.im(A[k + 1, k + 1]))
            if s < eps * norm:
                s = norm
            if abs(A[k + 1, k]) < eps * s:
                break
            k += 1
        if k + 1 < n1:
            A[k + 1, k] = 0
            n0 = k + 1
            its = 0
            if n0 + 1 >= n1:
                n0 = 0
                n1 = k + 1
                if n1 < 2:
                    return
        else:
            if its % 30 == 10:
                shift = A[n1 - 1, n1 - 2]
            elif its % 30 == 20:
                shift = abs(A[n1 - 1, n1 - 2])
            elif its % 30 == 29:
                shift = norm
            else:
                t = A[n1 - 2, n1 - 2] + A[n1 - 1, n1 - 1]
                s = (A[n1 - 1, n1 - 1] - A[n1 - 2, n1 - 2]) ** 2 + 4 * A[n1 - 1, n1 - 2] * A[n1 - 2, n1 - 1]
                if ctx.re(s) > 0:
                    s = ctx.sqrt(s)
                else:
                    s = ctx.sqrt(-s) * 1j
                a = (t + s) / 2
                b = (t - s) / 2
                if abs(A[n1 - 1, n1 - 1] - a) > abs(A[n1 - 1, n1 - 1] - b):
                    shift = b
                else:
                    shift = a
            its += 1
            totalits += 1
            qr_step(ctx, n0, n1, A, Q, shift)
            if its > maxits:
                raise RuntimeError('qr: failed to converge after %d steps' % its)