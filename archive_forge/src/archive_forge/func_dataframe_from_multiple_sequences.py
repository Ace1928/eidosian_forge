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
def dataframe_from_multiple_sequences(x_values, y_values):
    """
   Converts a set of multiple sequences (eg: time series), stored as a 2 dimensional
   numpy array into a pandas dataframe that can be plotted by datashader.
   The pandas dataframe eventually contains two columns ('x' and 'y') with the data.
   Each time series is separated by a row of NaNs.
   Discussion at: https://github.com/bokeh/datashader/issues/286#issuecomment-334619499

   x_values: 1D numpy array with the values to be plotted on the x axis (eg: time)
   y_values: 2D numpy array with the sequences to be plotted of shape (num sequences X length of
             each sequence)

   """
    x = np.zeros(x_values.shape[0] + 1)
    x[-1] = np.nan
    x[:-1] = x_values
    x = np.tile(x, y_values.shape[0])
    y = np.zeros((y_values.shape[0], y_values.shape[1] + 1))
    y[:, -1] = np.nan
    y[:, :-1] = y_values
    return pd.DataFrame({'x': x, 'y': y.flatten()})