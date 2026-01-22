from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
@pytest.mark.parametrize('freq_expected', freq_expected)
def test_freq_to_period(freq_expected):
    freq, expected = freq_expected
    assert_equal(tools.freq_to_period(freq), expected)
    assert_equal(tools.freq_to_period(to_offset(freq)), expected)