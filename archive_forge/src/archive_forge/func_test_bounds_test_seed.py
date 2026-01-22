from typing import NamedTuple
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest
from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl.model import (
from statsmodels.tsa.deterministic import DeterministicProcess
@pytest.mark.parametrize('seed', [None, np.random.RandomState(0), 0, [1, 2], np.random.default_rng([1, 2])])
def test_bounds_test_seed(seed):
    mod = UECM(dane_data.lrm, 3, dane_data[['lry', 'ibo', 'ide']], {'lry': 1, 'ibo': 3, 'ide': 2})
    res = mod.fit()
    bounds_result = res.bounds_test(case=3, asymptotic=False, seed=seed, nsim=10000)
    assert (bounds_result.p_values >= 0.0).all()
    assert (bounds_result.p_values <= 1.0).all()
    assert (bounds_result.crit_vals > 0.0).all().all()