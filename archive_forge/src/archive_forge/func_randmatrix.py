from ..libmp.backend import xrange
import warnings
def randmatrix(ctx, m, n=None, min=0, max=1, **kwargs):
    """
        Create a random m x n matrix.

        All values are >= min and <max.
        n defaults to m.

        Example:
        >>> from mpmath import randmatrix
        >>> randmatrix(2) # doctest:+SKIP
        matrix(
        [['0.53491598236191806', '0.57195669543302752'],
         ['0.85589992269513615', '0.82444367501382143']])
        """
    if not n:
        n = m
    A = ctx.matrix(m, n, **kwargs)
    for i in xrange(m):
        for j in xrange(n):
            A[i, j] = ctx.rand() * (max - min) + min
    return A