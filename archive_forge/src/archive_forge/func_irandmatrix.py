from mpmath import mp
from mpmath import libmp
def irandmatrix(n, range=10):
    """
    random matrix with integer entries
    """
    A = mp.matrix(n, n)
    for i in xrange(n):
        for j in xrange(n):
            A[i, j] = int((2 * mp.rand() - 1) * range)
    return A