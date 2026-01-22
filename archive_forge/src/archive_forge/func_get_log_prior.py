import warnings
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy
import numpy as np
import pandas as pd
from scipy.fftpack import next_fast_len
from scipy.interpolate import CubicSpline
from scipy.stats.mstats import mquantiles
from xarray import apply_ufunc
from .. import _log
from ..utils import conditional_jit, conditional_vect, conditional_dask
from .density_utils import histogram as _histogram
def get_log_prior(idata, var_names=None):
    """Retrieve the log prior dataarray of a given variable."""
    if not hasattr(idata, 'log_prior'):
        raise TypeError('log prior not found in inference data object')
    if var_names is None:
        var_names = list(idata.log_prior.data_vars)
    return idata.log_prior[var_names]