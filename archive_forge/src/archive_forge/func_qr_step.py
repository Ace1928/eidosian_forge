from ..libmp.backend import xrange
def qr_step(ctx, n0, n1, A, Q, shift):
    """
    This subroutine executes a single implicitly shifted QR step applied to an
    upper Hessenberg matrix A. Given A and shift as input, first an QR
    decomposition is calculated:

      Q R = A - shift * 1 .

    The output is then following matrix:

      R Q + shift * 1

    parameters:
      n0, n1    (input) Two integers which specify the submatrix A[n0:n1,n0:n1]
                on which this subroutine operators. The subdiagonal elements
                to the left and below this submatrix must be deflated (i.e. zero).
                following restriction is imposed: n1>=n0+2
      A         (input/output) On input, A is an upper Hessenberg matrix.
                On output, A is replaced by "R Q + shift * 1"
      Q         (input/output) The parameter Q is multiplied by the unitary matrix
                Q arising from the QR decomposition. Q can also be false, in which
                case the unitary matrix Q is not computated.
      shift     (input) a complex number specifying the shift. idealy close to an
                eigenvalue of the bottemmost part of the submatrix A[n0:n1,n0:n1].

    references:
      Stoer, Bulirsch - Introduction to Numerical Analysis.
      Kresser : Numerical Methods for General and Structured Eigenvalue Problems
    """
    n = A.rows
    c = A[n0, n0] - shift
    s = A[n0 + 1, n0]
    v = ctx.hypot(ctx.hypot(ctx.re(c), ctx.im(c)), ctx.hypot(ctx.re(s), ctx.im(s)))
    if v == 0:
        v = 1
        c = 1
        s = 0
    else:
        c /= v
        s /= v
    cc = ctx.conj(c)
    cs = ctx.conj(s)
    for k in xrange(n0, n):
        x = A[n0, k]
        y = A[n0 + 1, k]
        A[n0, k] = cc * x + cs * y
        A[n0 + 1, k] = c * y - s * x
    for k in xrange(min(n1, n0 + 3)):
        x = A[k, n0]
        y = A[k, n0 + 1]
        A[k, n0] = c * x + s * y
        A[k, n0 + 1] = cc * y - cs * x
    if not isinstance(Q, bool):
        for k in xrange(n):
            x = Q[k, n0]
            y = Q[k, n0 + 1]
            Q[k, n0] = c * x + s * y
            Q[k, n0 + 1] = cc * y - cs * x
    for j in xrange(n0, n1 - 2):
        c = A[j + 1, j]
        s = A[j + 2, j]
        v = ctx.hypot(ctx.hypot(ctx.re(c), ctx.im(c)), ctx.hypot(ctx.re(s), ctx.im(s)))
        if v == 0:
            A[j + 1, j] = 0
            v = 1
            c = 1
            s = 0
        else:
            A[j + 1, j] = v
            c /= v
            s /= v
        A[j + 2, j] = 0
        cc = ctx.conj(c)
        cs = ctx.conj(s)
        for k in xrange(j + 1, n):
            x = A[j + 1, k]
            y = A[j + 2, k]
            A[j + 1, k] = cc * x + cs * y
            A[j + 2, k] = c * y - s * x
        for k in xrange(0, min(n1, j + 4)):
            x = A[k, j + 1]
            y = A[k, j + 2]
            A[k, j + 1] = c * x + s * y
            A[k, j + 2] = cc * y - cs * x
        if not isinstance(Q, bool):
            for k in xrange(0, n):
                x = Q[k, j + 1]
                y = Q[k, j + 2]
                Q[k, j + 1] = c * x + s * y
                Q[k, j + 2] = cc * y - cs * x