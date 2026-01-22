from ..libmp.backend import xrange
from .eigen import defun
@defun
def eigsy(ctx, A, eigvals_only=False, overwrite_a=False):
    """
    This routine solves the (ordinary) eigenvalue problem for a real symmetric
    square matrix A. Given A, an orthogonal matrix Q is calculated which
    diagonalizes A:

          Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) is a diagonal matrix whose diagonal is E.
    ' denotes the transpose.

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

          A Q[:,i] = E[i] Q[:,i]


    input:

      A: real matrix of format (n,n) which is symmetric
         (i.e. A=A' or A[i,j]=A[j,i])

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: orthogonal matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

          E          if eigvals_only is true
         (E, Q)      if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, 2], [2, 0]])
      >>> E = mp.eigsy(A, eigvals_only = True)
      >>> print(E)
      [-1.0]
      [ 4.0]

      >>> A = mp.matrix([[1, 2], [2, 3]])
      >>> E, Q = mp.eigsy(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eighe, eigh, eig
    """
    if not overwrite_a:
        A = A.copy()
    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)
    if eigvals_only:
        r_sy_tridiag(ctx, A, d, e, calc_ev=False)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        r_sy_tridiag(ctx, A, d, e, calc_ev=True)
        tridiag_eigen(ctx, d, e, A)
        return (d, A)