import scipy.linalg._interpolative_backend as _backend
import numpy as np
import sys
def reconstruct_interp_matrix(idx, proj):
    """
    Reconstruct interpolation matrix from ID.

    The interpolation matrix can be reconstructed from the ID indices and
    coefficients `idx` and `proj`, respectively, as::

        P = numpy.hstack([numpy.eye(proj.shape[0]), proj])[:,numpy.argsort(idx)]

    The original matrix can then be reconstructed from its skeleton matrix `B`
    via::

        numpy.dot(B, P)

    See also :func:`reconstruct_matrix_from_id` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_reconint` and
        :func:`_backend.idz_reconint`.

    Parameters
    ----------
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolation matrix.
    """
    if _is_real(proj):
        return _backend.idd_reconint(idx + 1, proj)
    else:
        return _backend.idz_reconint(idx + 1, proj)