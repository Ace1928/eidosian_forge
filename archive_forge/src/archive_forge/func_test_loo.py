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
@pytest.mark.parametrize('scale', ['log', 'negative_log', 'deviance'])
@pytest.mark.parametrize('multidim', (True, False))
def test_loo(centered_eight, multidim_models, scale, multidim):
    """Test approximate leave one out criterion calculation"""
    if multidim:
        assert loo(multidim_models.model_1, scale=scale) is not None
        loo_pointwise = loo(multidim_models.model_1, pointwise=True, scale=scale)
    else:
        assert loo(centered_eight, scale=scale) is not None
        loo_pointwise = loo(centered_eight, pointwise=True, scale=scale)
    assert loo_pointwise is not None
    assert 'loo_i' in loo_pointwise
    assert 'pareto_k' in loo_pointwise
    assert 'scale' in loo_pointwise