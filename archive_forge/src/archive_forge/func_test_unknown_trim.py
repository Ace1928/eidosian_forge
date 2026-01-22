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
def test_unknown_trim(self):
    assert_raises(ValueError, stattools.lagmat, self.macro_df, 3, trim='unknown', use_pandas=True)
    assert_raises(ValueError, stattools.lagmat, self.macro_df.values, 3, trim='unknown')