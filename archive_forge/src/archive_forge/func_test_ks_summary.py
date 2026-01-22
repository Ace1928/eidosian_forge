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
def test_ks_summary(self):
    """Instead of psislw data, this test uses fake data."""
    pareto_tail_indices = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
    with pytest.warns(UserWarning):
        summary = ks_summary(pareto_tail_indices)
    assert summary is not None
    pareto_tail_indices2 = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.6])
    with pytest.warns(UserWarning):
        summary2 = ks_summary(pareto_tail_indices2)
    assert summary2 is not None