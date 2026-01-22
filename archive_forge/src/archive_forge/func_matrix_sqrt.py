import numpy as np
def matrix_sqrt(mat, inverse=False, full=False, nullspace=False, threshold=1e-15):
    """matrix square root for symmetric matrices

    Usage is for decomposing a covariance function S into a square root R
    such that

        R' R = S if inverse is False, or
        R' R = pinv(S) if inverse is True

    Parameters
    ----------
    mat : array_like, 2-d square
        symmetric square matrix for which square root or inverse square
        root is computed.
        There is no checking for whether the matrix is symmetric.
        A warning is issued if some singular values are negative, i.e.
        below the negative of the threshold.
    inverse : bool
        If False (default), then the matrix square root is returned.
        If inverse is True, then the matrix square root of the inverse
        matrix is returned.
    full : bool
        If full is False (default, then the square root has reduce number
        of rows if the matrix is singular, i.e. has singular values below
        the threshold.
    nullspace : bool
        If nullspace is true, then the matrix square root of the null space
        of the matrix is returned.
    threshold : float
        Singular values below the threshold are dropped.

    Returns
    -------
    msqrt : ndarray
        matrix square root or square root of inverse matrix.
    """
    u, s, v = np.linalg.svd(mat)
    if np.any(s < -threshold):
        import warnings
        warnings.warn('some singular values are negative')
    if not nullspace:
        mask = s > threshold
        s[s < threshold] = 0
    else:
        mask = s < threshold
        s[s > threshold] = 0
    sqrt_s = np.sqrt(s[mask])
    if inverse:
        sqrt_s = 1 / np.sqrt(s[mask])
    if full:
        b = np.dot(u[:, mask], np.dot(np.diag(sqrt_s), v[mask]))
    else:
        b = np.dot(np.diag(sqrt_s), v[mask])
    return b