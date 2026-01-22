import warnings
import numpy as np
from .._shared.utils import check_nD
from ..color import gray2rgb
from ..util import img_as_float
from ._texture import _glcm_loop, _local_binary_pattern, _multiblock_lbp
def local_binary_pattern(image, P, R, method='default'):
    """Compute the local binary patterns (LBP) of an image.

    LBP is a visual descriptor often used in texture classification.

    Parameters
    ----------
    image : (M, N) array
        2D grayscale image.
    P : int
        Number of circularly symmetric neighbor set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : str {'default', 'ror', 'uniform', 'nri_uniform', 'var'}, optional
        Method to determine the pattern:

        ``default``
            Original local binary pattern which is grayscale invariant but not
            rotation invariant.
        ``ror``
            Extension of default pattern which is grayscale invariant and
            rotation invariant.
        ``uniform``
            Uniform pattern which is grayscale invariant and rotation
            invariant, offering finer quantization of the angular space.
            For details, see [1]_.
        ``nri_uniform``
            Variant of uniform pattern which is grayscale invariant but not
            rotation invariant. For details, see [2]_ and [3]_.
        ``var``
            Variance of local image texture (related to contrast)
            which is rotation invariant but not grayscale invariant.

    Returns
    -------
    output : (M, N) array
        LBP image.

    References
    ----------
    .. [1] T. Ojala, M. Pietikainen, T. Maenpaa, "Multiresolution gray-scale
           and rotation invariant texture classification with local binary
           patterns", IEEE Transactions on Pattern Analysis and Machine
           Intelligence, vol. 24, no. 7, pp. 971-987, July 2002
           :DOI:`10.1109/TPAMI.2002.1017623`
    .. [2] T. Ahonen, A. Hadid and M. Pietikainen. "Face recognition with
           local binary patterns", in Proc. Eighth European Conf. Computer
           Vision, Prague, Czech Republic, May 11-14, 2004, pp. 469-481, 2004.
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851
           :DOI:`10.1007/978-3-540-24670-1_36`
    .. [3] T. Ahonen, A. Hadid and M. Pietikainen, "Face Description with
           Local Binary Patterns: Application to Face Recognition",
           IEEE Transactions on Pattern Analysis and Machine Intelligence,
           vol. 28, no. 12, pp. 2037-2041, Dec. 2006
           :DOI:`10.1109/TPAMI.2006.244`
    """
    check_nD(image, 2)
    methods = {'default': ord('D'), 'ror': ord('R'), 'uniform': ord('U'), 'nri_uniform': ord('N'), 'var': ord('V')}
    if np.issubdtype(image.dtype, np.floating):
        warnings.warn('Applying `local_binary_pattern` to floating-point images may give unexpected results when small numerical differences between adjacent pixels are present. It is recommended to use this function with images of integer dtype.')
    image = np.ascontiguousarray(image, dtype=np.float64)
    output = _local_binary_pattern(image, P, R, methods[method.lower()])
    return output