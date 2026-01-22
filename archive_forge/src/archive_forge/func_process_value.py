import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
@staticmethod
def process_value(value):
    """
        Homogenize the input *value* for easy and efficient normalization.

        *value* can be a scalar or sequence.

        Returns
        -------
        result : masked array
            Masked array with the same shape as *value*.
        is_scalar : bool
            Whether *value* is a scalar.

        Notes
        -----
        Float dtypes are preserved; integer types with two bytes or smaller are
        converted to np.float32, and larger types are converted to np.float64.
        Preserving float32 when possible, and using in-place operations,
        greatly improves speed for large arrays.
        """
    is_scalar = not np.iterable(value)
    if is_scalar:
        value = [value]
    dtype = np.min_scalar_type(value)
    if np.issubdtype(dtype, np.integer) or dtype.type is np.bool_:
        dtype = np.promote_types(dtype, np.float32)
    mask = np.ma.getmask(value)
    data = np.asarray(value)
    result = np.ma.array(data, mask=mask, dtype=dtype, copy=True)
    return (result, is_scalar)