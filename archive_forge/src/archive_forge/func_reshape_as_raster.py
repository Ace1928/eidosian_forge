from collections import OrderedDict
from itertools import zip_longest
import logging
import warnings
import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.transform import guard_transform
def reshape_as_raster(arr):
    """Returns the array in a raster order
    by swapping the axes order from (rows, columns, bands)
    to (bands, rows, columns)

    Parameters
    ----------
    arr : array-like in the image form of (rows, columns, bands)
        image to reshape
    """
    im = np.transpose(arr, [2, 0, 1])
    return im