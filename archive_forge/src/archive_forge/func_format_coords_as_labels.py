import importlib
import warnings
from typing import Any, Dict
import matplotlib as mpl
import numpy as np
import packaging
from matplotlib.colors import to_hex
from scipy.stats import mode, rankdata
from scipy.interpolate import CubicSpline
from ..rcparams import rcParams
from ..stats.density_utils import kde
from ..stats import hdi
def format_coords_as_labels(dataarray, skip_dims=None):
    """Format 1d or multi-d dataarray coords as strings.

    Parameters
    ----------
    dataarray : xarray.DataArray
        DataArray whose coordinates will be converted to labels.
    skip_dims : str of list_like, optional
        Dimensions whose values should not be included in the labels
    """
    if skip_dims is None:
        coord_labels = dataarray.coords.to_index()
    else:
        coord_labels = dataarray.coords.to_index().droplevel(skip_dims).drop_duplicates()
    coord_labels = coord_labels.values
    if isinstance(coord_labels[0], tuple):
        fmt = ', '.join(['{}' for _ in coord_labels[0]])
        coord_labels[:] = [fmt.format(*x) for x in coord_labels]
    else:
        coord_labels[:] = [f'{s}' for s in coord_labels]
    return coord_labels