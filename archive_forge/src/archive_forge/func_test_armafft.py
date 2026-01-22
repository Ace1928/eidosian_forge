from statsmodels.compat.pandas import QUARTER_END
import datetime as dt
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.sandbox.tsa.fftarma import ArmaFft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import (
from statsmodels.tsa.tests.results import results_arma_acf
from statsmodels.tsa.tests.results.results_process import (
@pytest.mark.parametrize('ar', arlist)
@pytest.mark.parametrize('ma', malist)
def test_armafft(ar, ma):
    nfreq = 20
    w = np.linspace(0, np.pi, nfreq, endpoint=False)
    arma = ArmaFft(ar, ma, 20)
    ac1 = arma.invpowerspd(1024)[:10]
    ac2 = arma.acovf(10)[:10]
    assert_allclose(ac1, ac2, atol=1e-15, err_msg='acovf not equal for {}, {}'.format(ar, ma))