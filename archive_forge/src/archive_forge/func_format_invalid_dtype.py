import logging
import math
import os
import warnings
import numpy as np
import rasterio
from rasterio import warp
from rasterio._base import DatasetBase
from rasterio._features import _shapes, _sieve, _rasterize, _bounds
from rasterio.dtypes import validate_dtype, can_cast_dtype, get_minimum_dtype, _getnpdtype
from rasterio.enums import MergeAlg
from rasterio.env import ensure_env, GDALVersion
from rasterio.errors import ShapeSkipWarning
from rasterio.rio.helpers import coords
from rasterio.transform import Affine
from rasterio.transform import IDENTITY, guard_transform
from rasterio.windows import Window
def format_invalid_dtype(param):
    return '{0} dtype must be one of: {1}'.format(param, ', '.join(valid_dtypes))