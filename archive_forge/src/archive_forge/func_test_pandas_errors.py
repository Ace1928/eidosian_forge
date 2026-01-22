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
def test_pandas_errors(self):
    assert_raises(ValueError, stattools.lagmat, self.macro_df, 3, trim='none', use_pandas=True)
    assert_raises(ValueError, stattools.lagmat, self.macro_df, 3, trim='backward', use_pandas=True)
    assert_raises(ValueError, stattools.lagmat, self.series, 3, trim='none', use_pandas=True)
    assert_raises(ValueError, stattools.lagmat, self.series, 3, trim='backward', use_pandas=True)