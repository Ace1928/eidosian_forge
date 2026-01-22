from itertools import chain
from operator import add
import numpy as np
from ._haar import haar_like_feature_coord_wrapper
from ._haar import haar_like_feature_wrapper
from ..color import gray2rgb
from ..draw import rectangle
from ..util import img_as_float
def draw_haar_like_feature(image, r, c, width, height, feature_coord, color_positive_block=(1.0, 0.0, 0.0), color_negative_block=(0.0, 1.0, 0.0), alpha=0.5, max_n_features=None, rng=None):
    """Visualization of Haar-like features.

    Parameters
    ----------
    image : (M, N) ndarray
        The region of an integral image for which the features need to be
        computed.
    r : int
        Row-coordinate of top left corner of the detection window.
    c : int
        Column-coordinate of top left corner of the detection window.
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_coord : ndarray of list of tuples or None, optional
        The array of coordinates to be extracted. This is useful when you want
        to recompute only a subset of features. In this case `feature_type`
        needs to be an array containing the type of each feature, as returned
        by :func:`haar_like_feature_coord`. By default, all coordinates are
        computed.
    color_positive_block : tuple of 3 floats
        Floats specifying the color for the positive block. Corresponding
        values define (R, G, B) values. Default value is red (1, 0, 0).
    color_negative_block : tuple of 3 floats
        Floats specifying the color for the negative block Corresponding values
        define (R, G, B) values. Default value is blue (0, 1, 0).
    alpha : float
        Value in the range [0, 1] that specifies opacity of visualization. 1 -
        fully transparent, 0 - opaque.
    max_n_features : int, default=None
        The maximum number of features to be returned.
        By default, all features are returned.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.

        The rng is used when generating a set of features smaller than
        the total number of available features.

    Returns
    -------
    features : (M, N), ndarray
        An image in which the different features will be added.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.feature import haar_like_feature_coord
    >>> from skimage.feature import draw_haar_like_feature
    >>> feature_coord, _ = haar_like_feature_coord(2, 2, 'type-4')
    >>> image = draw_haar_like_feature(np.zeros((2, 2)),
    ...                                0, 0, 2, 2,
    ...                                feature_coord,
    ...                                max_n_features=1)
    >>> image
    array([[[0. , 0.5, 0. ],
            [0.5, 0. , 0. ]],
    <BLANKLINE>
           [[0.5, 0. , 0. ],
            [0. , 0.5, 0. ]]])

    """
    rng = np.random.default_rng(rng)
    color_positive_block = np.asarray(color_positive_block, dtype=np.float64)
    color_negative_block = np.asarray(color_negative_block, dtype=np.float64)
    if max_n_features is None:
        feature_coord_ = feature_coord
    else:
        feature_coord_ = rng.choice(feature_coord, size=max_n_features, replace=False)
    output = np.copy(image)
    if len(image.shape) < 3:
        output = gray2rgb(image)
    output = img_as_float(output)
    for coord in feature_coord_:
        for idx_rect, rect in enumerate(coord):
            coord_start, coord_end = rect
            coord_start = tuple(map(add, coord_start, [r, c]))
            coord_end = tuple(map(add, coord_end, [r, c]))
            rr, cc = rectangle(coord_start, coord_end)
            if (idx_rect + 1) % 2 == 0:
                new_value = (1 - alpha) * output[rr, cc] + alpha * color_positive_block
            else:
                new_value = (1 - alpha) * output[rr, cc] + alpha * color_negative_block
            output[rr, cc] = new_value
    return output