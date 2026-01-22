from mpmath import mp
from mpmath import libmp
def test_eighe_irandmatrix():
    N = 4
    R = 4
    for a in xrange(10):
        A = irandmatrix(N, R) + 1j * irandmatrix(N, R)
        for i in xrange(0, N):
            A[i, i] = mp.re(A[i, i])
            for j in xrange(i + 1, N):
                A[j, i] = mp.conj(A[i, j])
        run_eighe(A)