from ..libmp.backend import xrange
from .eigen import defun
def c_he_tridiag_0(ctx, A, D, E, T):
    """
    This routine transforms a complex hermitian matrix A to a real symmetric
    tridiagonal matrix T using an unitary similarity transformation:
          Q' * A * Q = T     (here ' denotes the hermitian matrix transpose,
                              i.e. transposition und conjugation).
    The unitary matrix Q is build up from Householder reflectors and
    an unitary diagonal matrix.

    parameters:
      A         (input/output) On input, A contains the complex hermitian matrix
                of dimension (n,n). On output, A contains the unitary matrix Q
                in compressed form.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix.

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      T         (output) complex array of length n, contains a unitary diagonal
                matrix.

    This routine is a python translation (in slightly modified form) of the fortran
    routine htridi.f in the software library EISPACK (see netlib.org) which itself
    is a complex version of the algol procedure tred1 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """
    n = A.rows
    T[n - 1] = 1
    for i in xrange(n - 1, 0, -1):
        scale = 0
        for k in xrange(0, i):
            scale += abs(ctx.re(A[k, i])) + abs(ctx.im(A[k, i]))
        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale
        if scale == 0 or ctx.isinf(scale_inv):
            E[i] = 0
            D[i] = 0
            T[i - 1] = 1
            continue
        if i == 1:
            F = A[i - 1, i]
            f = abs(F)
            E[i] = f
            D[i] = 0
            if f != 0:
                T[i - 1] = T[i] * F / f
            else:
                T[i - 1] = T[i]
            continue
        H = 0
        for k in xrange(0, i):
            A[k, i] *= scale_inv
            rr = ctx.re(A[k, i])
            ii = ctx.im(A[k, i])
            H += rr * rr + ii * ii
        F = A[i - 1, i]
        f = abs(F)
        G = ctx.sqrt(H)
        H += G * f
        E[i] = scale * G
        if f != 0:
            F = F / f
            TZ = -T[i] * F
            G *= F
        else:
            TZ = -T[i]
        A[i - 1, i] += G
        F = 0
        for j in xrange(0, i):
            A[i, j] = A[j, i] / H
            G = 0
            for k in xrange(0, j + 1):
                G += ctx.conj(A[k, j]) * A[k, i]
            for k in xrange(j + 1, i):
                G += A[j, k] * A[k, i]
            T[j] = G / H
            F += ctx.conj(T[j]) * A[j, i]
        HH = F / (2 * H)
        for j in xrange(0, i):
            F = A[j, i]
            G = T[j] - HH * F
            T[j] = G
            for k in xrange(0, j + 1):
                A[k, j] -= ctx.conj(F) * T[k] + ctx.conj(G) * A[k, i]
        T[i - 1] = TZ
        D[i] = H
    for i in xrange(1, n):
        E[i - 1] = E[i]
    E[n - 1] = 0
    D[0] = 0
    for i in xrange(0, n):
        zw = D[i]
        D[i] = ctx.re(A[i, i])
        A[i, i] = zw