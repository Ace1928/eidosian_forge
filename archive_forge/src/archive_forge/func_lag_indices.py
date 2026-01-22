from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def lag_indices(self, lag):
    """return the index array for lagged values

        Warning: if k is larger then the number of observations for an
        individual, then no values for that individual are returned.

        TODO: for the unbalanced case, I should get the same truncation for
        the array with lag=0. From the return of lag_idx we would not know
        which individual is missing.

        TODO: do I want the full equivalent of lagmat in tsa?
        maxlag or lag or lags.

        not tested yet
        """
    lag_idx = np.asarray(self.groupidx)[:, 1] - lag
    mask_ok = lag <= lag_idx
    return lag_idx[mask_ok]