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
@pytest.fixture(scope='module')
def multivariable_log_likelihood(centered_eight):
    centered_eight = centered_eight.copy()
    new_arr = DataArray(np.zeros(centered_eight.log_likelihood['obs'].values.shape), dims=['chain', 'draw', 'school'], coords=centered_eight.log_likelihood.coords)
    centered_eight.log_likelihood['decoy'] = new_arr
    return centered_eight