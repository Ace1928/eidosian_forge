import warnings
import numpy as np
from .._shared.utils import check_nD
from ..color import gray2rgb
from ..util import img_as_float
from ._texture import _glcm_loop, _local_binary_pattern, _multiblock_lbp
def multiblock_lbp(int_image, r, c, width, height):
    """Multi-block local binary pattern (MB-LBP).

    The features are calculated similarly to local binary patterns (LBPs),
    (See :py:meth:`local_binary_pattern`) except that summed blocks are
    used instead of individual pixel values.

    MB-LBP is an extension of LBP that can be computed on multiple scales
    in constant time using the integral image. Nine equally-sized rectangles
    are used to compute a feature. For each rectangle, the sum of the pixel
    intensities is computed. Comparisons of these sums to that of the central
    rectangle determine the feature, similarly to LBP.

    Parameters
    ----------
    int_image : (N, M) array
        Integral image.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of the 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of the 9 equal rectangles that will be used to compute
        a feature.

    Returns
    -------
    output : int
        8-bit MB-LBP feature descriptor.

    References
    ----------
    .. [1] L. Zhang, R. Chu, S. Xiang, S. Liao, S.Z. Li. "Face Detection Based
           on Multi-Block LBP Representation", In Proceedings: Advances in
           Biometrics, International Conference, ICB 2007, Seoul, Korea.
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
           :DOI:`10.1007/978-3-540-74549-5_2`
    """
    int_image = np.ascontiguousarray(int_image, dtype=np.float32)
    lbp_code = _multiblock_lbp(int_image, r, c, width, height)
    return lbp_code