import warnings
import numpy as np
from .._shared.utils import check_nD
from ..color import gray2rgb
from ..util import img_as_float
from ._texture import _glcm_loop, _local_binary_pattern, _multiblock_lbp
Multi-block local binary pattern visualization.

    Blocks with higher sums are colored with alpha-blended white rectangles,
    whereas blocks with lower sums are colored alpha-blended cyan. Colors
    and the `alpha` parameter can be changed.

    Parameters
    ----------
    image : ndarray of float or uint
        Image on which to visualize the pattern.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of 9 equal rectangles that will be used to compute
        a feature.
    lbp_code : int
        The descriptor of feature to visualize. If not provided, the
        descriptor with 0 value will be used.
    color_greater_block : tuple of 3 floats
        Floats specifying the color for the block that has greater
        intensity value. They should be in the range [0, 1].
        Corresponding values define (R, G, B) values. Default value
        is white (1, 1, 1).
    color_greater_block : tuple of 3 floats
        Floats specifying the color for the block that has greater intensity
        value. They should be in the range [0, 1]. Corresponding values define
        (R, G, B) values. Default value is cyan (0, 0.69, 0.96).
    alpha : float
        Value in the range [0, 1] that specifies opacity of visualization.
        1 - fully transparent, 0 - opaque.

    Returns
    -------
    output : ndarray of float
        Image with MB-LBP visualization.

    References
    ----------
    .. [1] L. Zhang, R. Chu, S. Xiang, S. Liao, S.Z. Li. "Face Detection Based
           on Multi-Block LBP Representation", In Proceedings: Advances in
           Biometrics, International Conference, ICB 2007, Seoul, Korea.
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
           :DOI:`10.1007/978-3-540-74549-5_2`
    