from __future__ import annotations
from numbers import Number
from math import log10
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from xarray import DataArray, Dataset
from .utils import Dispatcher, ngjit, calc_res, calc_bbox, orient_array, \
from .utils import get_indices, dshape_from_pandas, dshape_from_dask
from .utils import Expr # noqa (API import)
from .resampling import resample_2d, resample_2d_distributed
from . import reductions as rd
def compute_scale_and_translate(self, range, n):
    """Compute the scale and translate parameters for a linear transformation
        ``output = s * input + t``, mapping from data space to axis space.

        Parameters
        ----------
        range : tuple
            A tuple representing the range ``[min, max]`` along the axis, in
            data space. Both min and max are inclusive.
        n : int
            The number of bins along the axis.

        Returns
        -------
        s, t : floats
        """
    start, end = map(self.mapper, range)
    s = n / (end - start)
    t = -start * s
    return (s, t)