import numpy as np
from scipy.stats import pearsonr
from .._shared.utils import check_shape_equality, as_binary_ndarray
def intersection_coeff(image0_mask, image1_mask, mask=None):
    """Fraction of a channel's segmented binary mask that overlaps with a
    second channel's segmented binary mask.

    Parameters
    ----------
    image0_mask : (M, N) ndarray of dtype bool
        Image mask of channel A.
    image1_mask : (M, N) ndarray of dtype bool
        Image mask of channel B.
        Must have same dimensions as `image0_mask`.
    mask : (M, N) ndarray of dtype bool, optional
        Only `image0_mask` and `image1_mask` pixels within this region of
        interest
        mask are included in the calculation.
        Must have same dimensions as `image0_mask`.

    Returns
    -------
    Intersection coefficient, float
        Fraction of `image0_mask` that overlaps with `image1_mask`.

    """
    image0_mask = as_binary_ndarray(image0_mask, variable_name='image0_mask')
    image1_mask = as_binary_ndarray(image1_mask, variable_name='image1_mask')
    if mask is not None:
        mask = as_binary_ndarray(mask, variable_name='mask')
        check_shape_equality(image0_mask, image1_mask, mask)
        image0_mask = image0_mask[mask]
        image1_mask = image1_mask[mask]
    else:
        check_shape_equality(image0_mask, image1_mask)
    nonzero_image0 = np.count_nonzero(image0_mask)
    if nonzero_image0 == 0:
        return 0
    nonzero_joint = np.count_nonzero(np.logical_and(image0_mask, image1_mask))
    return nonzero_joint / nonzero_image0