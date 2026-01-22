from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
def test_correct_nobs():
    mdata = macrodata.load_pandas().data
    dates = mdata[['year', 'quarter']].astype(int).astype(str)
    quarterly = dates['year'] + 'Q' + dates['quarter']
    quarterly = dates_from_str(quarterly)
    mdata = mdata[['realgdp', 'realcons', 'realinv']]
    mdata.index = pd.DatetimeIndex(quarterly)
    data = np.log(mdata).diff().dropna()
    data.index.freq = data.index.inferred_freq
    data_exog = pd.DataFrame(index=data.index)
    data_exog['exovar1'] = np.random.normal(size=data_exog.shape[0])
    model = VAR(endog=data, exog=data_exog)
    results = model.fit(maxlags=1)
    irf = results.irf_resim(orth=False, repl=100, steps=10, seed=1, burn=100, cum=False)
    assert irf.shape == (100, 11, 3, 3)