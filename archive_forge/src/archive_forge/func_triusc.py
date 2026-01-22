import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def triusc(x, dims, offset=0):
    """
    Scales the strictly lower triangular part of the 's' components of x 
    by 0.5.
    """
    m = dims['l'] + sum(dims['q']) + sum([k ** 2 for k in dims['s']])
    ind = offset + dims['l'] + sum(dims['q'])
    for mk in dims['s']:
        for j in range(1, mk):
            blas.scal(0.5, x, offset=ind + mk * (j - 1) + j, n=mk - j)
        ind += mk ** 2