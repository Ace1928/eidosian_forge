from copy import deepcopy
import numpy as np
import pytest
from numpy.testing import (
from scipy.special import logsumexp
from scipy.stats import linregress, norm, halfcauchy
from xarray import DataArray, Dataset
from xarray_einstats.stats import XrContinuousRV
from ...data import concat, convert_to_inference_data, from_dict, load_arviz_data
from ...rcparams import rcParams
from ...stats import (
from ...stats.stats import _gpinv
from ...stats.stats_utils import get_log_likelihood
from ..helpers import check_multiple_attrs, multidim_models  # pylint: disable=unused-import
def test_hdi_multimodal_multivars():
    size = 2500000
    var1 = np.concatenate((np.random.normal(-4, 1, size), np.random.normal(2, 0.5, size)))
    var2 = np.random.normal(8, 1, size * 2)
    sample = Dataset({'var1': (('chain', 'draw'), var1[np.newaxis, :]), 'var2': (('chain', 'draw'), var2[np.newaxis, :])}, coords={'chain': [0], 'draw': np.arange(size * 2)})
    intervals = hdi(sample, multimodal=True)
    assert_array_almost_equal(intervals.var1, [[-5.8, -2.2], [0.9, 3.1]], 1)
    assert_array_almost_equal(intervals.var2, [[6.1, 9.9], [np.nan, np.nan]], 1)