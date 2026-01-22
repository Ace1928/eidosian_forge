from mpmath import mp
from mpmath import libmp
def test_svd_r_rand():
    for i in xrange(5):
        full = mp.rand() > 0.5
        m = 1 + int(mp.rand() * 10)
        n = 1 + int(mp.rand() * 10)
        A = 2 * mp.randmatrix(m, n) - 1
        if mp.rand() > 0.5:
            A *= 10
            for x in xrange(m):
                for y in xrange(n):
                    A[x, y] = int(A[x, y])
        run_svd_r(A, full_matrices=full, verbose=False)