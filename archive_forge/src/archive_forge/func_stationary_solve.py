import numpy as np
def stationary_solve(r, b):
    """
    Solve a linear system for a Toeplitz correlation matrix.

    A Toeplitz correlation matrix represents the covariance of a
    stationary series with unit variance.

    Parameters
    ----------
    r : array_like
        A vector describing the coefficient matrix.  r[0] is the first
        band next to the diagonal, r[1] is the second band, etc.
    b : array_like
        The right-hand side for which we are solving, i.e. we solve
        Tx = b and return b, where T is the Toeplitz coefficient matrix.

    Returns
    -------
    The solution to the linear system.
    """
    db = r[0:1]
    dim = b.ndim
    if b.ndim == 1:
        b = b[:, None]
    x = b[0:1, :]
    for j in range(1, len(b)):
        rf = r[0:j][::-1]
        a = (b[j, :] - np.dot(rf, x)) / (1 - np.dot(rf, db[::-1]))
        z = x - np.outer(db[::-1], a)
        x = np.concatenate((z, a[None, :]), axis=0)
        if j == len(b) - 1:
            break
        rn = r[j]
        a = (rn - np.dot(rf, db)) / (1 - np.dot(rf, db[::-1]))
        z = db - a * db[::-1]
        db = np.concatenate((z, np.r_[a]))
    if dim == 1:
        x = x[:, 0]
    return x