import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
def calc_moments(timeseries_file, moment):
    """Returns nth moment (3 for skewness, 4 for kurtosis) of timeseries
    (list of values; one per timeseries).

    Keyword arguments:
    timeseries_file -- text file with white space separated timepoints in rows

    """
    import scipy.stats as stats
    timeseries = np.genfromtxt(timeseries_file)
    m2 = stats.moment(timeseries, 2, axis=0)
    m3 = stats.moment(timeseries, moment, axis=0)
    zero = m2 == 0
    return np.where(zero, 0, m3 / m2 ** (moment / 2.0))