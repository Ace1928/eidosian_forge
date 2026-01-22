from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def interpret_data(data, colnames=None, rownames=None):
    """
    Convert passed data structure to form required by estimation classes

    Parameters
    ----------
    data : array_like
    colnames : sequence or None
        May be part of data structure
    rownames : sequence or None

    Returns
    -------
    (values, colnames, rownames) : (homogeneous ndarray, list)
    """
    if isinstance(data, np.ndarray):
        values = np.asarray(data)
        if colnames is None:
            colnames = ['Y_%d' % i for i in range(values.shape[1])]
    elif is_data_frame(data):
        data = data.dropna()
        values = data.values
        colnames = data.columns
        rownames = data.index
    else:
        raise TypeError('Cannot handle input type {typ}'.format(typ=type(data).__name__))
    if not isinstance(colnames, list):
        colnames = list(colnames)
    if len(colnames) != values.shape[1]:
        raise ValueError('length of colnames does not match number of columns in data')
    if rownames is not None and len(rownames) != len(values):
        raise ValueError('length of rownames does not match number of rows in data')
    return (values, colnames, rownames)