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
@conditional_jit(nopython=True)
def stats_variance_1d(data, ddof=0):
    a_a, b_b = (0, 0)
    for i in data:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / len(data) - (a_a / len(data)) ** 2
    var = var * (len(data) / (len(data) - ddof))
    return var