from itertools import product, combinations_with_replacement, permutations
import re
import pickle
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy.stats import norm  # type: ignore[attr-defined]
from scipy.stats._axis_nan_policy import _masked_arrays_2_sentinel_arrays
from scipy._lib._util import AxisError
def nan_policy_1d(hypotest, data1d, unpacker, *args, n_outputs=2, nan_policy='raise', paired=False, _no_deco=True, **kwds):
    if nan_policy == 'raise':
        for sample in data1d:
            if np.any(np.isnan(sample)):
                raise ValueError('The input contains nan values')
    elif nan_policy == 'propagate' and hypotest not in override_propagate_funcs:
        for sample in data1d:
            if np.any(np.isnan(sample)):
                return np.full(n_outputs, np.nan)
    elif nan_policy == 'omit':
        if not paired:
            data1d = [sample[~np.isnan(sample)] for sample in data1d]
        else:
            nan_mask = np.isnan(data1d[0])
            for sample in data1d[1:]:
                nan_mask = np.logical_or(nan_mask, np.isnan(sample))
            data1d = [sample[~nan_mask] for sample in data1d]
    return unpacker(hypotest(*data1d, *args, _no_deco=_no_deco, **kwds))