from functools import reduce
import numpy as np
def to_matvec(transform):
    """Split a transform into its matrix and vector components

    The transformation must be represented in homogeneous coordinates and is
    split into its rotation matrix and translation vector components.

    Parameters
    ----------
    transform : array-like
        NxM transform matrix in homogeneous coordinates representing an affine
        transformation from an (N-1)-dimensional space to an (M-1)-dimensional
        space. An example is a 4x4 transform representing rotations and
        translations in 3 dimensions. A 4x3 matrix can represent a
        2-dimensional plane embedded in 3 dimensional space.

    Returns
    -------
    matrix : (N-1, M-1) array
        Matrix component of `transform`
    vector : (M-1,) array
        Vector component of `transform`

    See Also
    --------
    from_matvec

    Examples
    --------
    >>> aff = np.diag([2, 3, 4, 1])
    >>> aff[:3,3] = [9, 10, 11]
    >>> to_matvec(aff)
    (array([[2, 0, 0],
           [0, 3, 0],
           [0, 0, 4]]), array([ 9, 10, 11]))
    """
    transform = np.asarray(transform)
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    matrix = transform[0:ndimin, 0:ndimout]
    vector = transform[0:ndimin, ndimout]
    return (matrix, vector)