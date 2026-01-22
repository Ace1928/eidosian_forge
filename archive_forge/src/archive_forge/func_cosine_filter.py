import os
import os.path as op
from collections import OrderedDict
from itertools import chain
import nibabel as nb
import numpy as np
from numpy.polynomial import Legendre
from .. import config, logging
from ..external.due import BibTeX
from ..interfaces.base import (
from ..utils.misc import normalize_mc_params
def cosine_filter(data, timestep, period_cut, remove_mean=True, axis=-1, failure_mode='error'):
    datashape = data.shape
    timepoints = datashape[axis]
    if datashape[0] == 0 and failure_mode != 'error':
        return (data, np.array([]))
    data = data.reshape((-1, timepoints))
    frametimes = timestep * np.arange(timepoints)
    X = _full_rank(_cosine_drift(period_cut, frametimes))[0]
    non_constant_regressors = X[:, :-1] if X.shape[1] > 1 else np.array([])
    betas = np.linalg.lstsq(X, data.T)[0]
    if not remove_mean:
        X = X[:, :-1]
        betas = betas[:-1]
    residuals = data - X.dot(betas).T
    return (residuals.reshape(datashape), non_constant_regressors)