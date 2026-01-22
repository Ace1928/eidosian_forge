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
@pytest.mark.parametrize('func', [loo, waic])
def test_multidimensional_log_likelihood(func):
    llm = np.random.rand(4, 23, 15, 2)
    ll1 = llm.reshape(4, 23, 15 * 2)
    statsm = Dataset(dict(log_likelihood=DataArray(llm, dims=['chain', 'draw', 'a', 'b'])))
    stats1 = Dataset(dict(log_likelihood=DataArray(ll1, dims=['chain', 'draw', 'v'])))
    post = Dataset(dict(mu=DataArray(np.random.rand(4, 23, 2), dims=['chain', 'draw', 'v'])))
    dsm = convert_to_inference_data(statsm, group='sample_stats')
    ds1 = convert_to_inference_data(stats1, group='sample_stats')
    dsp = convert_to_inference_data(post, group='posterior')
    dsm = concat(dsp, dsm)
    ds1 = concat(dsp, ds1)
    frm = func(dsm)
    fr1 = func(ds1)
    assert all((fr1[key] == frm[key] for key in fr1.index if key not in {'loo_i', 'waic_i', 'pareto_k'}))
    assert_array_almost_equal(frm[:4], fr1[:4])