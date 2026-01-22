from mpmath import mp
from mpmath import libmp
def test_eigsy_irandmatrix():
    N = 4
    R = 4
    for a in xrange(10):
        A = irandmatrix(N, R)
        for i in xrange(0, N):
            for j in xrange(i + 1, N):
                A[j, i] = A[i, j]
        run_eigsy(A)