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
def test_lagmat2ds_numpy(self):
    data = self.macro_df
    npdata = data.values
    lagmat = stattools.lagmat2ds(npdata, 2)
    expected = self._prepare_expected(npdata, 2)
    assert_array_equal(lagmat, expected)
    lagmat = stattools.lagmat2ds(npdata[:, :2], 3)
    expected = self._prepare_expected(npdata[:, :2], 3)
    assert_array_equal(lagmat, expected)
    npdata = self.series.values
    lagmat = stattools.lagmat2ds(npdata, 5)
    expected = self._prepare_expected(npdata[:, None], 5)
    assert_array_equal(lagmat, expected)