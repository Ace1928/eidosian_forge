import numpy as np
import pandas as pd
from scipy import stats
@property
def se_mean(self):
    """The standard deviation of the predicted mean"""
    ndim = self._var_pred_mean.ndim
    if ndim == 1:
        values = np.sqrt(self._var_pred_mean)
    elif ndim == 3:
        values = np.sqrt(self._var_pred_mean.T.diagonal())
    else:
        raise NotImplementedError('var_pre_mean must be 1 or 3 dim')
    return self._wrap_pandas(values, 'mean_se')