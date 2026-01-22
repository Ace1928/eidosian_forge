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
@pytest.mark.parametrize('missing_groups', (None, 'posterior', 'prior'))
def test_summary_groups(centered_eight, missing_groups):
    if missing_groups == 'posterior':
        centered_eight = deepcopy(centered_eight)
        del centered_eight.posterior
    elif missing_groups == 'prior':
        centered_eight = deepcopy(centered_eight)
        del centered_eight.posterior
        del centered_eight.prior
    if missing_groups == 'prior':
        with pytest.warns(UserWarning):
            summary_df = summary(centered_eight)
    else:
        summary_df = summary(centered_eight)
    assert summary_df.shape