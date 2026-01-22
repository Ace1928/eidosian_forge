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
@pytest.mark.parametrize('var_names_expected', ((None, 10), ('mu', 1), (['mu', 'tau'], 2)))
def test_summary_var_names(centered_eight, var_names_expected):
    var_names, expected = var_names_expected
    summary_df = summary(centered_eight, var_names=var_names)
    assert len(summary_df.index) == expected