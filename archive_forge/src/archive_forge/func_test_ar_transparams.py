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
def test_ar_transparams():
    arr = np.array([-1000.0, -100.0, -10.0, 1.0, 0.0, 1.0, 10.0, 100.0, 1000.0])
    assert not np.isnan(tools._ar_transparams(arr)).any()