import os
import numpy as np
import packaging
import pandas as pd
import pytest
import scipy
from numpy.testing import assert_almost_equal
from ...data import from_cmdstan, load_arviz_data
from ...rcparams import rcParams
from ...sel_utils import xarray_var_iter
from ...stats import bfmi, ess, mcse, rhat
from ...stats.diagnostics import (
@pytest.mark.parametrize('relative', (True, False))
def test_effective_sample_size_too_many_probs(self, relative):
    with pytest.raises(ValueError):
        ess(np.random.randn(4, 100), method='local', prob=[0.1, 0.2, 0.9], relative=relative)