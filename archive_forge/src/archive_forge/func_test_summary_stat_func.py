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
@pytest.mark.parametrize('stat_funcs', [[np.var], {'var': np.var, 'var2': lambda x: np.var(x) ** 2}])
def test_summary_stat_func(centered_eight, stat_funcs):
    arviz_summary = summary(centered_eight, stat_funcs=stat_funcs)
    assert arviz_summary is not None
    assert hasattr(arviz_summary, 'var')