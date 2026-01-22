from mpmath import mp
from mpmath import libmp
def run_schur(A, verbose=0):
    if verbose > 1:
        print('original matrix (schur):\n', A)
    n = A.rows
    Q, R = mp.schur(A)
    if verbose > 1:
        print('Q:\n', Q)
        print('R:\n', R)
    B = Q * R * Q.transpose_conj()
    C = Q * Q.transpose_conj()
    eps = mp.exp(0.8 * mp.log(mp.eps))
    err0 = 0
    for x in xrange(n):
        for y in xrange(n):
            err0 += abs(A[y, x] - B[y, x])
    err0 /= n * n
    err1 = 0
    for x in xrange(n):
        for y in xrange(n):
            if x == y:
                C[y, x] -= 1
            err1 += abs(C[y, x])
    err1 /= n * n
    err2 = 0
    for x in xrange(n):
        for y in xrange(x + 1, n):
            err2 += abs(R[y, x])
    if verbose > 0:
        print('difference (S):', err0, err1, err2)
    if verbose > 1:
        print('B:\n', B)
    assert err0 < eps
    assert err1 < eps
    assert err2 == 0