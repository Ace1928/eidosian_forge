import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def lagged_groups(x, lag, groupidx):
    """
    assumes sorted by time, groupidx is tuple of start and end values
    not optimized, just to get a working version, loop over groups
    """
    out0 = []
    out_lagged = []
    for l, u in groupidx:
        if l + lag < u:
            out0.append(x[l + lag:u])
            out_lagged.append(x[l:u - lag])
    if out0 == []:
        raise ValueError('all groups are empty taking lags')
    return (np.vstack(out0), np.vstack(out_lagged))