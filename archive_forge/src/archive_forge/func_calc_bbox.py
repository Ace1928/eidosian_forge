from __future__ import annotations
import os
import re
from inspect import getmro
import numba as nb
import numpy as np
import pandas as pd
from toolz import memoize
from xarray import DataArray
import dask.dataframe as dd
import datashader.datashape as datashape
def calc_bbox(xs, ys, res):
    """Calculate the bounding box of a raster, and return it in a four-element
    tuple: (xmin, ymin, xmax, ymax). This calculation assumes the raster is
    uniformly sampled (equivalent to a flat-earth assumption, for geographic
    data) so that an affine transform (using the "Augmented Matrix" approach)
    suffices:
    https://en.wikipedia.org/wiki/Affine_transformation#Augmented_matrix

    Parameters
    ----------
    xs : numpy.array
        1D NumPy array of floats representing the x-values of a raster. This
        likely originated from an xarray.DataArray or xarray.Dataset object
        (xr.open_rasterio).
    ys : numpy.array
        1D NumPy array of floats representing the y-values of a raster. This
        likely originated from an xarray.DataArray or xarray.Dataset object
        (xr.open_rasterio).
    res : tuple
        Two-tuple (int, int) which includes x and y resolutions (aka "grid/cell
        sizes"), respectively.
    """
    xbound = xs.max() if res[0] < 0 else xs.min()
    ybound = ys.min() if res[1] < 0 else ys.max()
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    Ab = np.array([[res[0], 0.0, xbound], [0.0, -res[1], ybound], [0.0, 0.0, 1.0]])
    for x_, y_ in [(0, 0), (0, len(ys)), (len(xs), 0), (len(xs), len(ys))]:
        x, y, _ = np.dot(Ab, np.array([x_, y_, 1.0]))
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    xpad, ypad = (res[0] / 2.0, res[1] / 2.0)
    return (xmin - xpad, ymin + ypad, xmax - xpad, ymax + ypad)