import inspect
import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, warn
from .._shared.version_requirements import require
from ..exposure import histogram
from ..filters._multiotsu import (
from ..transform import integral_image
from ..util import dtype_limits
from ._sparse import _correlate_sparse, _validate_window_size
def threshold_multiotsu(image=None, classes=3, nbins=256, *, hist=None):
    """Generate `classes`-1 threshold values to divide gray levels in `image`,
    following Otsu's method for multiple classes.

    The threshold values are chosen to maximize the total sum of pairwise
    variances between the thresholded graylevel classes. See Notes and [1]_
    for more details.

    Either image or hist must be provided. If hist is provided, the actual
    histogram of the image is ignored.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray, optional
        Grayscale input image.
    classes : int, optional
        Number of classes to be thresholded, i.e. the number of resulting
        regions.
    nbins : int, optional
        Number of bins used to calculate the histogram. This value is ignored
        for integer arrays.
    hist : array, or 2-tuple of arrays, optional
        Histogram from which to determine the threshold, and optionally a
        corresponding array of bin center intensities. If no hist provided,
        this function will compute it from the image (see notes).

    Returns
    -------
    thresh : array
        Array containing the threshold values for the desired classes.

    Raises
    ------
    ValueError
         If ``image`` contains less grayscale value then the desired
         number of classes.

    Notes
    -----
    This implementation relies on a Cython function whose complexity
    is :math:`O\\left(\\frac{Ch^{C-1}}{(C-1)!}\\right)`, where :math:`h`
    is the number of histogram bins and :math:`C` is the number of
    classes desired.

    If no hist is given, this function will make use of
    `skimage.exposure.histogram`, which behaves differently than
    `np.histogram`. While both allowed, use the former for consistent
    behaviour.

    The input image must be grayscale.

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    .. [2] Tosa, Y., "Multi-Otsu Threshold", a java plugin for ImageJ.
           Available at:
           <http://imagej.net/plugins/download/Multi_OtsuThreshold.java>

    Examples
    --------
    >>> from skimage.color import label2rgb
    >>> from skimage import data
    >>> image = data.camera()
    >>> thresholds = threshold_multiotsu(image)
    >>> regions = np.digitize(image, bins=thresholds)
    >>> regions_colorized = label2rgb(regions)
    """
    if image is not None and image.ndim > 2 and (image.shape[-1] in (3, 4)):
        warn(f'threshold_multiotsu is expected to work correctly only for grayscale images; image shape {image.shape} looks like that of an RGB image.')
    prob, bin_centers = _validate_image_histogram(image, hist, nbins, normalize=True)
    prob = prob.astype('float32', copy=False)
    nvalues = np.count_nonzero(prob)
    if nvalues < classes:
        msg = f'After discretization into bins, the input image has only {nvalues} different values. It cannot be thresholded in {classes} classes. If there are more unique values before discretization, try increasing the number of bins (`nbins`).'
        raise ValueError(msg)
    elif nvalues == classes:
        thresh_idx = np.flatnonzero(prob)[:-1]
    else:
        try:
            thresh_idx = _get_multiotsu_thresh_indices_lut(prob, classes - 1)
        except MemoryError:
            thresh_idx = _get_multiotsu_thresh_indices(prob, classes - 1)
    thresh = bin_centers[thresh_idx]
    return thresh