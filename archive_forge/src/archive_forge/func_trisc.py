import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def trisc(x, dims, offset=0):
    """
    Sets upper triangular part of the 's' components of x equal to zero
    and scales the strictly lower triangular part by 2.0.
    """
    m = dims['l'] + sum(dims['q']) + sum([k ** 2 for k in dims['s']])
    ind = offset + dims['l'] + sum(dims['q'])
    for mk in dims['s']:
        for j in range(1, mk):
            blas.scal(0.0, x, n=mk - j, inc=mk, offset=ind + j * (mk + 1) - 1)
            blas.scal(2.0, x, offset=ind + mk * (j - 1) + j, n=mk - j)
        ind += mk ** 2