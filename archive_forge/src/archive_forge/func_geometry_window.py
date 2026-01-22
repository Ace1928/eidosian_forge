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
def geometry_window(dataset, shapes, pad_x=0, pad_y=0, north_up=None, rotated=None, pixel_precision=None, boundless=False):
    """Calculate the window within the raster that fits the bounds of
    the geometry plus optional padding.  The window is the outermost
    pixel indices that contain the geometry (floor of offsets, ceiling
    of width and height).

    If shapes do not overlap raster, a WindowError is raised.

    Parameters
    ----------
    dataset : dataset object opened in 'r' mode
        Raster for which the mask will be created.
    shapes : iterable over geometries.
        A geometry is a GeoJSON-like object or implements the geo
        interface.  Must be in same coordinate system as dataset.
    pad_x : float
        Amount of padding (as fraction of raster's x pixel size) to add
        to left and right side of bounds.
    pad_y : float
        Amount of padding (as fraction of raster's y pixel size) to add
        to top and bottom of bounds.
    north_up : optional
        This parameter is ignored since version 1.2.1. A deprecation
        warning will be emitted in 1.3.0.
    rotated : optional
        This parameter is ignored since version 1.2.1. A deprecation
        warning will be emitted in 1.3.0.
    pixel_precision : int or float, optional
        Number of places of rounding precision or absolute precision for
        evaluating bounds of shapes.
    boundless : bool, optional
        Whether to allow a boundless window or not.

    Returns
    -------
    rasterio.windows.Window

    """
    all_bounds = [bounds(shape, transform=~dataset.transform) for shape in shapes]
    cols = [x for left, bottom, right, top in all_bounds for x in (left - pad_x, right + pad_x, right + pad_x, left - pad_x)]
    rows = [y for left, bottom, right, top in all_bounds for y in (top - pad_y, top - pad_y, bottom + pad_y, bottom + pad_y)]
    row_start, row_stop = (int(math.floor(min(rows))), int(math.ceil(max(rows))))
    col_start, col_stop = (int(math.floor(min(cols))), int(math.ceil(max(cols))))
    window = Window(col_off=col_start, row_off=row_start, width=max(col_stop - col_start, 0.0), height=max(row_stop - row_start, 0.0))
    raster_window = Window(0, 0, dataset.width, dataset.height)
    if not boundless:
        window = window.intersection(raster_window)
    return window