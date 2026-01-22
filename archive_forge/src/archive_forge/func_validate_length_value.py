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
def validate_length_value(instance, attribute, value):
    if value and value < 0:
        raise ValueError('Number of columns or rows must be non-negative')