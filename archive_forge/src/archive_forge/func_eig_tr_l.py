from ..libmp.backend import xrange
def eig_tr_l(ctx, A):
    """
    This routine calculates the left eigenvectors of an upper right triangular matrix.

    input:
      A      an upper right triangular matrix

    output:
      EL     a matrix whose rows form the left eigenvectors of A

    return value:  EL
    """
    n = A.rows
    EL = ctx.eye(n)
    eps = ctx.eps
    unfl = ctx.ldexp(ctx.one, -ctx.prec * 30)
    smlnum = unfl * (n / eps)
    simin = 1 / ctx.sqrt(eps)
    rmax = 1
    for i in xrange(0, n - 1):
        s = A[i, i]
        smin = max(eps * abs(s), smlnum)
        for j in xrange(i + 1, n):
            r = 0
            for k in xrange(i, j):
                r += EL[i, k] * A[k, j]
            t = A[j, j] - s
            if abs(t) < smin:
                t = smin
            r = -r / t
            EL[i, j] = r
            rmax = max(rmax, abs(r))
            if rmax > simin:
                for k in xrange(i, j + 1):
                    EL[i, k] /= rmax
                rmax = 1
        if rmax != 1:
            for k in xrange(i, n):
                EL[i, k] /= rmax
    return EL