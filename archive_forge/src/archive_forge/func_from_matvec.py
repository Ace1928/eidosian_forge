from functools import reduce
import numpy as np
def from_matvec(matrix, vector=None):
    """Combine a matrix and vector into an homogeneous affine

    Combine a rotation / scaling / shearing matrix and translation vector into
    a transform in homogeneous coordinates.

    Parameters
    ----------
    matrix : array-like
        An NxM array representing the the linear part of the transform.
        A transform from an M-dimensional space to an N-dimensional space.
    vector : None or array-like, optional
        None or an (N,) array representing the translation. None corresponds to
        an (N,) array of zeros.

    Returns
    -------
    xform : array
        An (N+1, M+1) homogeneous transform matrix.

    See Also
    --------
    to_matvec

    Examples
    --------
    >>> from_matvec(np.diag([2, 3, 4]), [9, 10, 11])
    array([[ 2,  0,  0,  9],
           [ 0,  3,  0, 10],
           [ 0,  0,  4, 11],
           [ 0,  0,  0,  1]])

    The `vector` argument is optional:

    >>> from_matvec(np.diag([2, 3, 4]))
    array([[2, 0, 0, 0],
           [0, 3, 0, 0],
           [0, 0, 4, 0],
           [0, 0, 0, 1]])
    """
    matrix = np.asarray(matrix)
    nin, nout = matrix.shape
    t = np.zeros((nin + 1, nout + 1), matrix.dtype)
    t[0:nin, 0:nout] = matrix
    t[nin, nout] = 1.0
    if vector is not None:
        t[0:nin, nout] = vector
    return t