from ..libmp.backend import xrange
def eig_tr_r(ctx, A):
    """
    This routine calculates the right eigenvectors of an upper right triangular matrix.

    input:
      A      an upper right triangular matrix

    output:
      ER     a matrix whose columns form the right eigenvectors of A

    return value: ER
    """
    n = A.rows
    ER = ctx.eye(n)
    eps = ctx.eps
    unfl = ctx.ldexp(ctx.one, -ctx.prec * 30)
    smlnum = unfl * (n / eps)
    simin = 1 / ctx.sqrt(eps)
    rmax = 1
    for i in xrange(1, n):
        s = A[i, i]
        smin = max(eps * abs(s), smlnum)
        for j in xrange(i - 1, -1, -1):
            r = 0
            for k in xrange(j + 1, i + 1):
                r += A[j, k] * ER[k, i]
            t = A[j, j] - s
            if abs(t) < smin:
                t = smin
            r = -r / t
            ER[j, i] = r
            rmax = max(rmax, abs(r))
            if rmax > simin:
                for k in xrange(j, i + 1):
                    ER[k, i] /= rmax
                rmax = 1
        if rmax != 1:
            for k in xrange(0, i + 1):
                ER[k, i] /= rmax
    return ER