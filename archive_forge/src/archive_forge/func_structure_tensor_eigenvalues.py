import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def structure_tensor_eigenvalues(A_elems):
    """Compute eigenvalues of structure tensor.

    Parameters
    ----------
    A_elems : list of ndarray
        The upper-diagonal elements of the structure tensor, as returned
        by `structure_tensor`.

    Returns
    -------
    ndarray
        The eigenvalues of the structure tensor, in decreasing order. The
        eigenvalues are the leading dimension. That is, the coordinate
        [i, j, k] corresponds to the ith-largest eigenvalue at position (j, k).

    Examples
    --------
    >>> from skimage.feature import structure_tensor
    >>> from skimage.feature import structure_tensor_eigenvalues
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 1
    >>> A_elems = structure_tensor(square, sigma=0.1, order='rc')
    >>> structure_tensor_eigenvalues(A_elems)[0]
    array([[0., 0., 0., 0., 0.],
           [0., 2., 4., 2., 0.],
           [0., 4., 0., 4., 0.],
           [0., 2., 4., 2., 0.],
           [0., 0., 0., 0., 0.]])

    See also
    --------
    structure_tensor
    """
    return _symmetric_compute_eigenvalues(A_elems)