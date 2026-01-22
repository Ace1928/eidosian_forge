from . import select
from . import utils
from scipy import sparse
import numpy as np
import pandas as pd
import scipy.signal
def gene_variability(data, kernel_size=0.005, smooth=5, return_means=False):
    """Measure the variability of each gene in a dataset.

    Variability is computed as the deviation from
    the rolling median of the mean-variance curve

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    kernel_size : float or int, optional (default: 0.005)
        Width of rolling median window. If a float between 0 and 1, the width is given
        by kernel_size * data.shape[1]. Otherwise should be an odd integer
    smooth : int, optional (default: 5)
        Amount of smoothing to apply to the median filter
    return_means : boolean, optional (default: False)
        If True, return the gene means

    Returns
    -------
    variability : list-like, shape=[n_samples]
        Variability for each gene
    """
    columns = data.columns if isinstance(data, pd.DataFrame) else None
    data = utils.to_array_or_spmatrix(data)
    if isinstance(data, sparse.dia_matrix):
        data = data.tocsc()
    data_std = utils.matrix_std(data, axis=0) ** 2
    data_mean = utils.toarray(data.mean(axis=0)).flatten()
    if kernel_size < 1:
        kernel_size = 2 * (int(kernel_size * len(data_std)) // 2) + 1
    order = np.argsort(data_mean)
    data_std_med = np.empty_like(data_std)
    data_std_order = data_std[order]
    data_std_order = np.r_[data_std_order[kernel_size::-1], data_std_order, data_std_order[:-kernel_size:-1]]
    medfilt = scipy.signal.medfilt(data_std_order, kernel_size=kernel_size)[kernel_size:-kernel_size]
    for i in range(smooth):
        medfilt = np.r_[(medfilt[1:] + medfilt[:-1]) / 2, medfilt[-1]]
    data_std_med[order] = medfilt
    result = data_std - data_std_med
    if columns is not None:
        result = pd.Series(result, index=columns, name='variability')
        data_mean = pd.Series(data_mean, index=columns, name='mean')
    if return_means:
        result = (result, data_mean)
    return result