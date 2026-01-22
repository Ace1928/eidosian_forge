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
def psens_data():
    non_centered_eight = load_arviz_data('non_centered_eight')
    post = non_centered_eight.posterior
    log_prior = {'mu': XrContinuousRV(norm, 0, 5).logpdf(post['mu']), 'tau': XrContinuousRV(halfcauchy, scale=5).logpdf(post['tau']), 'theta_t': XrContinuousRV(norm, 0, 1).logpdf(post['theta_t'])}
    non_centered_eight.add_groups({'log_prior': log_prior})
    return non_centered_eight