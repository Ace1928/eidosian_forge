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
@pytest.mark.matplotlib
@pytest.mark.smoke
@pytest.mark.parametrize('trend', ['n', 'c', 'ct'])
@pytest.mark.parametrize('seasonal', [True, False])
def test_ardl_smoke_plots(data, seasonal, trend, close_figures):
    from matplotlib.figure import Figure
    mod = ARDL(data.y, 3, trend=trend, seasonal=seasonal)
    res = mod.fit()
    fig = res.plot_diagnostics()
    assert isinstance(fig, Figure)
    fig = res.plot_predict(end=100)
    assert isinstance(fig, Figure)
    fig = res.plot_predict(end=75, alpha=None, in_sample=False)
    assert isinstance(fig, Figure)
    assert isinstance(res.summary(), Summary)