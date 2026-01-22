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
def test_ardl_parameter_names(data):
    mod = ARDL(data.y, 2, data.x, 2, causal=True, trend='c')
    expected = ['const', 'lrm.L1', 'lrm.L2', 'lry.L1', 'lry.L2', 'ibo.L1', 'ibo.L2', 'ide.L1', 'ide.L2']
    assert mod.exog_names == expected
    mod = ARDL(np.asarray(data.y), 2, np.asarray(data.x), 2, causal=False, trend='ct')
    expected = ['const', 'trend', 'y.L1', 'y.L2', 'x0.L0', 'x0.L1', 'x0.L2', 'x1.L0', 'x1.L1', 'x1.L2', 'x2.L0', 'x2.L1', 'x2.L2']
    assert mod.exog_names == expected
    mod = ARDL(np.asarray(data.y), [2], np.asarray(data.x), None, causal=False, trend='n', seasonal=True, period=4)
    expected = ['s(1,4)', 's(2,4)', 's(3,4)', 's(4,4)', 'y.L2']
    assert mod.exog_names == expected