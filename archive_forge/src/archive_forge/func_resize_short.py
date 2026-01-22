import sys
import os
import random
import logging
import json
import warnings
from numbers import Number
import numpy as np
from .. import numpy as _mx_np  # pylint: disable=reimported
from ..base import numeric_types
from .. import ndarray as nd
from ..ndarray import _internal
from .. import io
from .. import recordio
from .. util import is_np_array
from ..ndarray.numpy import _internal as _npi
def resize_short(src, size, interp=2):
    """Resizes shorter edge to size.

    .. note:: `resize_short` uses OpenCV (not the CV2 Python library).
       MXNet must have been built with OpenCV for `resize_short` to work.

    Resizes the original image by setting the shorter edge to size
    and setting the longer edge accordingly.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    size : int
        The length to be set for the shorter edge.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.

    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.

    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> size = 640
    >>> new_image = mx.img.resize_short(image, size)
    >>> new_image
    <NDArray 2321x3482x3 @cpu(0)>
    """
    h, w, _ = src.shape
    if h > w:
        new_h, new_w = (size * h // w, size)
    else:
        new_h, new_w = (size, size * w // h)
    return imresize(src, new_w, new_h, interp=_get_interp_method(interp, (h, w, new_h, new_w)))