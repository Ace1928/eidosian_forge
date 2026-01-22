import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
def test_specifications():
    endog = np.arange(20).reshape(10, 2)
    exog = np.arange(10)
    exog2 = pd.Series(exog, index=pd.date_range('2000-01-01', '2009-01-01', freq='YS'))
    varmax.VARMAX(endog, exog=exog, order=(1, 0))
    varmax.VARMAX(endog, exog=exog2, order=(1, 0))