from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def t_statistic(X, Y):
    """Calculate Welch's t statistic.

    Assumes data of unequal number of samples and unequal variance

    Parameters
    ----------
    X : array-like, shape=[n_cells, n_genes]
    Y : array-like, shape=[m_cells, n_genes]

    Returns
    -------
    t_statistic : list-like, shape=[n_genes]
    """
    X, Y = _preprocess_test_matrices(X, Y)
    X_std = utils.matrix_std(X, axis=0)
    Y_std = utils.matrix_std(Y, axis=0)
    paired_std = np.sqrt(X_std ** 2 / X.shape[0] + Y_std ** 2 / Y.shape[0])
    return mean_difference(X, Y) / paired_std