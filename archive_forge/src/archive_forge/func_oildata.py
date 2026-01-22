from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
@pytest.fixture
def oildata():
    data = [111.0091346, 130.8284341, 141.2870879, 154.2277747, 162.7408654, 192.1664835, 240.7997253, 304.2173901, 384.0045673, 429.6621566, 359.3169299, 437.2518544, 468.4007898, 424.4353365, 487.9794299, 509.8284478, 506.3472527, 340.1842374, 240.258921, 219.0327876, 172.0746632, 252.5900922, 221.0710774, 276.5187735, 271.1479517, 342.6186005, 428.3558357, 442.3945534, 432.7851482, 437.2497186, 437.2091599, 445.3640981, 453.1950104, 454.409641, 422.3789058, 456.0371217, 440.3866047, 425.1943725, 486.2051735, 500.4290861, 521.2759092, 508.947617, 488.8888577, 509.870575, 456.7229123, 473.8166029, 525.9508706, 549.8338076, 542.3404698]
    return pd.Series(data, index=pd.date_range('1965', '2013', freq='YS'))