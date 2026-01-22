from itertools import chain
from operator import add
import numpy as np
from ._haar import haar_like_feature_coord_wrapper
from ._haar import haar_like_feature_wrapper
from ..color import gray2rgb
from ..draw import rectangle
from ..util import img_as_float
def haar_like_feature_coord(width, height, feature_type=None):
    """Compute the coordinates of Haar-like features.

    Parameters
    ----------
    width : int
        Width of the detection window.
    height : int
        Height of the detection window.
    feature_type : str or list of str or None, optional
        The type of feature to consider:

        - 'type-2-x': 2 rectangles varying along the x axis;
        - 'type-2-y': 2 rectangles varying along the y axis;
        - 'type-3-x': 3 rectangles varying along the x axis;
        - 'type-3-y': 3 rectangles varying along the y axis;
        - 'type-4': 4 rectangles varying along x and y axis.

        By default all features are extracted.

    Returns
    -------
    feature_coord : (n_features, n_rectangles, 2, 2), ndarray of list of tuple coord
        Coordinates of the rectangles for each feature.
    feature_type : (n_features,), ndarray of str
        The corresponding type for each feature.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.transform import integral_image
    >>> from skimage.feature import haar_like_feature_coord
    >>> feat_coord, feat_type = haar_like_feature_coord(2, 2, 'type-4')
    >>> feat_coord # doctest: +SKIP
    array([ list([[(0, 0), (0, 0)], [(0, 1), (0, 1)],
                  [(1, 1), (1, 1)], [(1, 0), (1, 0)]])], dtype=object)
    >>> feat_type
    array(['type-4'], dtype=object)

    """
    feature_type_ = _validate_feature_type(feature_type)
    feat_coord, feat_type = zip(*[haar_like_feature_coord_wrapper(width, height, feat_t) for feat_t in feature_type_])
    return (np.concatenate(feat_coord), np.hstack(feat_type))