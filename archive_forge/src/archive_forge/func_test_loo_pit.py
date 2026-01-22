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
@pytest.mark.parametrize('args', [{'y': 'obs'}, {'y': 'obs', 'y_hat': 'obs'}, {'y': 'arr', 'y_hat': 'obs'}, {'y': 'obs', 'y_hat': 'arr'}, {'y': 'arr', 'y_hat': 'arr'}, {'y': 'obs', 'y_hat': 'obs', 'log_weights': 'arr'}, {'y': 'arr', 'y_hat': 'obs', 'log_weights': 'arr'}, {'y': 'obs', 'y_hat': 'arr', 'log_weights': 'arr'}, {'idata': False}])
def test_loo_pit(centered_eight, args):
    y = args.get('y', None)
    y_hat = args.get('y_hat', None)
    log_weights = args.get('log_weights', None)
    y_arr = centered_eight.observed_data.obs
    y_hat_arr = centered_eight.posterior_predictive.obs.stack(__sample__=('chain', 'draw'))
    log_like = get_log_likelihood(centered_eight).stack(__sample__=('chain', 'draw'))
    n_samples = len(log_like.__sample__)
    ess_p = ess(centered_eight.posterior, method='mean')
    reff = np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / n_samples
    log_weights_arr = psislw(-log_like, reff=reff)[0]
    if args.get('idata', True):
        if y == 'arr':
            y = y_arr
        if y_hat == 'arr':
            y_hat = y_hat_arr
        if log_weights == 'arr':
            log_weights = log_weights_arr
        loo_pit_data = loo_pit(idata=centered_eight, y=y, y_hat=y_hat, log_weights=log_weights)
    else:
        loo_pit_data = loo_pit(idata=None, y=y_arr, y_hat=y_hat_arr, log_weights=log_weights_arr)
    assert np.all((loo_pit_data >= 0) & (loo_pit_data <= 1))