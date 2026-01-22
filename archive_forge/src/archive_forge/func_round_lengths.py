import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
def round_lengths(self, **kwds):
    """Return a copy with width and height rounded.

        Lengths are rounded to the nearest whole number. The offsets are
        not changed.

        Parameters
        ----------
        kwds : dict
            Collects keyword arguments that are no longer used.

        Returns
        -------
        Window

        """
    operator = lambda x: int(math.floor(x + 0.5))
    width = operator(self.width)
    height = operator(self.height)
    return Window(self.col_off, self.row_off, width, height)