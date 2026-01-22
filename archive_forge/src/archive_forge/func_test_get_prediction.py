from typing import NamedTuple
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest
from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl.model import (
from statsmodels.tsa.deterministic import DeterministicProcess
def test_get_prediction(data):
    res = ARDL(data.y, 3).fit()
    ar_res = AutoReg(data.y, 3).fit()
    pred = res.get_prediction(end='2020-01-01')
    ar_pred = ar_res.get_prediction(end='2020-01-01')
    assert_allclose(pred.predicted_mean, ar_pred.predicted_mean)
    assert_allclose(pred.var_pred_mean, ar_pred.var_pred_mean)