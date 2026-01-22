import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def sprod(x, y, dims, mnl=0, diag='N'):
    """
    The product x := (y o x).  If diag is 'D', the 's' part of y is 
    diagonal and only the diagonal is stored.
    """
    blas.tbmv(y, x, n=mnl + dims['l'], k=0, ldA=1)
    ind = mnl + dims['l']
    for m in dims['q']:
        dd = blas.dot(x, y, offsetx=ind, offsety=ind, n=m)
        blas.scal(y[ind], x, offset=ind + 1, n=m - 1)
        blas.axpy(y, x, alpha=x[ind], n=m - 1, offsetx=ind + 1, offsety=ind + 1)
        x[ind] = dd
        ind += m
    if diag == 'N':
        maxm = max([0] + dims['s'])
        A = matrix(0.0, (maxm, maxm))
        for m in dims['s']:
            blas.copy(x, A, offsetx=ind, n=m * m)
            for i in range(m - 1):
                symm(A, m)
                symm(y, m, offset=ind)
            blas.syr2k(A, y, x, alpha=0.5, n=m, k=m, ldA=m, ldB=m, ldC=m, offsetB=ind, offsetC=ind)
            ind += m * m
    else:
        ind2 = ind
        for m in dims['s']:
            for j in range(m):
                u = 0.5 * (y[ind2 + j:ind2 + m] + y[ind2 + j])
                blas.tbmv(u, x, n=m - j, k=0, ldA=1, offsetx=ind + j * (m + 1))
            ind += m * m
            ind2 += m