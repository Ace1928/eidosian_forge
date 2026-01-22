from . import utils
import numpy as np
import pandas as pd
import warnings
def check_numeric(data, dtype='float', copy=None, suppress_errors=False, default_fill_value=0.0):
    """Check a matrix contains only numeric data.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    dtype : str or `np.dtype`, optional (default: 'float')
        Data type to which to coerce the data
    copy : bool or None, optional (default: None)
        Copy the data before coercion. If None, default to
        False for all datatypes except pandas.SparseDataFrame
    suppress_errors : bool, optional (default: False)
        Suppress errors from non-numeric data
    default_fill_value : float
        If sparse, the default fill value

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Output data as numeric type

    Raises
    ------
    TypeError : if `data` cannot be coerced to `dtype`
    """
    if copy is None:
        copy = utils.is_SparseDataFrame(data)
    try:
        if utils.is_sparse_dataframe(data) and (not isinstance(dtype, pd.SparseDtype)):
            dtype = pd.SparseDtype(dtype, fill_value=default_fill_value)
        return data.astype(dtype, copy=copy)
    except TypeError as e:
        if utils.is_SparseDataFrame(data):
            if not copy:
                raise TypeError('pd.SparseDataFrame does not support copy=False. Please use copy=True.')
            else:
                return data.astype(dtype)
        else:
            raise e
    except ValueError:
        if suppress_errors:
            warnings.warn('Data is not numeric. Many scprep functions will not work.', RuntimeWarning)
            return data
        else:
            raise