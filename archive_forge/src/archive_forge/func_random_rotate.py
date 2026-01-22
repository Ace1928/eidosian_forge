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
def random_rotate(src, angle_limits, zoom_in=False, zoom_out=False):
    """Random rotates `src` by an angle included in angle limits.

    Parameters
    ----------
    src : NDArray
        Input image (format CHW) or batch of images (format NCHW),
        in both case is required a float32 data type.
    angle_limits: tuple
        Tuple of 2 elements containing the upper and lower limit
        for rotation angles in degree.
    zoom_in: bool
        If True input image(s) will be zoomed in a way so that no padding
        will be shown in the output result.
    zoom_out: bool
        If True input image(s) will be zoomed in a way so that the whole
        original image will be contained in the output result.
    Returns
    -------
    NDArray
        An `NDArray` containing the rotated image(s).
    """
    if src.ndim == 3:
        rotation_degrees = np.random.uniform(*angle_limits)
    else:
        n = src.shape[0]
        rotation_degrees = nd.array(np.random.uniform(*angle_limits, size=n))
    return imrotate(src, rotation_degrees, zoom_in=zoom_in, zoom_out=zoom_out)