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
def test_add_lag_noinsert(self):
    data = self.macro_df.values
    nddata = data.astype(float)
    lagmat = stattools.lagmat(nddata[:, 2], 3, trim='Both')
    results = np.column_stack((nddata[3:, :], lagmat))
    lag_data = tools.add_lag(data, self.realgdp_loc, 3, insert=False)
    assert_equal(lag_data, results)