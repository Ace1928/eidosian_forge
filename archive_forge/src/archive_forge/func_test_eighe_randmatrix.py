from mpmath import mp
from mpmath import libmp
def test_eighe_randmatrix():
    N = 5
    for a in xrange(10):
        A = 2 * mp.randmatrix(N, N) - 1 + 1j * (2 * mp.randmatrix(N, N) - 1)
        for i in xrange(0, N):
            A[i, i] = mp.re(A[i, i])
            for j in xrange(i + 1, N):
                A[j, i] = mp.conj(A[i, j])
        run_eighe(A)