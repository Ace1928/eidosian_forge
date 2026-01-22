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
def round_shape(self, **kwds):
    warnings.warn('round_shape is deprecated and will be removed in Rasterio 2.0.0.', RasterioDeprecationWarning)
    return self.round_lengths(**kwds)