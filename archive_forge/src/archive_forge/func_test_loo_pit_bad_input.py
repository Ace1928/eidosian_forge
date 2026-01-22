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
@pytest.mark.parametrize('input_type', ['idataarray', 'idatanone_ystr', 'yarr_yhatnone'])
def test_loo_pit_bad_input(centered_eight, input_type):
    """Test incompatible input combinations."""
    arr = np.random.random((8, 200))
    if input_type == 'idataarray':
        with pytest.raises(ValueError, match='type InferenceData or None'):
            loo_pit(idata=arr, y='obs')
    elif input_type == 'idatanone_ystr':
        with pytest.raises(ValueError, match='all 3.+must be array or DataArray'):
            loo_pit(idata=None, y='obs')
    elif input_type == 'yarr_yhatnone':
        with pytest.raises(ValueError, match='y_hat.+None.+y.+str'):
            loo_pit(idata=centered_eight, y=arr, y_hat=None)