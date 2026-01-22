import scipy.linalg._interpolative_backend as _backend
import numpy as np
import sys
def reconstruct_matrix_from_id(B, idx, proj):
    """
    Reconstruct matrix from its ID.

    A matrix `A` with skeleton matrix `B` and ID indices and coefficients `idx`
    and `proj`, respectively, can be reconstructed as::

        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

    See also :func:`reconstruct_interp_matrix` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_reconid` and
        :func:`_backend.idz_reconid`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Skeleton matrix.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Reconstructed matrix.
    """
    if _is_real(B):
        return _backend.idd_reconid(B, idx + 1, proj)
    else:
        return _backend.idz_reconid(B, idx + 1, proj)