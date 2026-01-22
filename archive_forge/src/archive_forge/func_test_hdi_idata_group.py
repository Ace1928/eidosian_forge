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
def test_hdi_idata_group(centered_eight):
    result_posterior = hdi(centered_eight, group='posterior', var_names='mu')
    result_prior = hdi(centered_eight, group='prior', var_names='mu')
    assert result_prior.sizes == {'hdi': 2}
    range_posterior = result_posterior.mu.values[1] - result_posterior.mu.values[0]
    range_prior = result_prior.mu.values[1] - result_prior.mu.values[0]
    assert range_posterior < range_prior