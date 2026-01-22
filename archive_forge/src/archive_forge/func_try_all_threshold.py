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
@require('matplotlib', '>=3.3')
def try_all_threshold(image, figsize=(8, 5), verbose=True):
    """Returns a figure comparing the outputs of different thresholding methods.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    figsize : tuple, optional
        Figure size (in inches).
    verbose : bool, optional
        Print function name for each method.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.

    Notes
    -----
    The following algorithms are used:

    * isodata
    * li
    * mean
    * minimum
    * otsu
    * triangle
    * yen

    Examples
    --------
    .. testsetup::
        >>> import pytest; _ = pytest.importorskip('matplotlib')

    >>> from skimage.data import text
    >>> fig, ax = try_all_threshold(text(), figsize=(10, 6), verbose=False)
    """

    def thresh(func):
        """
        A wrapper function to return a thresholded image.
        """

        def wrapper(im):
            return im > func(im)
        try:
            wrapper.__orifunc__ = func.__orifunc__
        except AttributeError:
            wrapper.__orifunc__ = func.__module__ + '.' + func.__name__
        return wrapper
    methods = OrderedDict({'Isodata': thresh(threshold_isodata), 'Li': thresh(threshold_li), 'Mean': thresh(threshold_mean), 'Minimum': thresh(threshold_minimum), 'Otsu': thresh(threshold_otsu), 'Triangle': thresh(threshold_triangle), 'Yen': thresh(threshold_yen)})
    return _try_all(image, figsize=figsize, methods=methods, verbose=verbose)