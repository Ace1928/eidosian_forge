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
@pytest.mark.parametrize('ic', ['loo', 'waic'])
def test_compare_multiple_obs(multivariable_log_likelihood, centered_eight, non_centered_eight, ic):
    compare_dict = {'centered_eight': centered_eight, 'non_centered_eight': non_centered_eight, 'problematic': multivariable_log_likelihood}
    with pytest.raises(TypeError, match='several log likelihood arrays'):
        get_log_likelihood(compare_dict['problematic'])
    with pytest.raises(TypeError, match='error in ELPD computation'):
        compare(compare_dict, ic=ic)
    assert compare(compare_dict, ic=ic, var_name='obs') is not None