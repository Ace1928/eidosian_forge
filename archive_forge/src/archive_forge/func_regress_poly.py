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
def regress_poly(degree, data, remove_mean=True, axis=-1, failure_mode='error'):
    """
    Returns data with degree polynomial regressed out.

    :param bool remove_mean: whether or not demean data (i.e. degree 0),
    :param int axis: numpy array axes along which regression is performed

    """
    IFLOGGER.debug('Performing polynomial regression on data of shape %s', str(data.shape))
    datashape = data.shape
    timepoints = datashape[axis]
    if datashape[0] == 0 and failure_mode != 'error':
        return (data, np.array([]))
    data = data.reshape((-1, timepoints))
    X = np.ones((timepoints, 1))
    for i in range(degree):
        polynomial_func = Legendre.basis(i + 1)
        value_array = np.linspace(-1, 1, timepoints)
        X = np.hstack((X, polynomial_func(value_array)[:, np.newaxis]))
    non_constant_regressors = X[:, :-1] if X.shape[1] > 1 else np.array([])
    betas = np.linalg.pinv(X).dot(data.T)
    if remove_mean:
        datahat = X.dot(betas).T
    else:
        datahat = X[:, 1:].dot(betas[1:, ...]).T
    regressed_data = data - datahat
    return (regressed_data.reshape(datashape), non_constant_regressors)