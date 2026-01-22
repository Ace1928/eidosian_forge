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
def test_ardl_order_keys_exceptions(data):
    with pytest.raises(ValueError, match='order dictionary contains keys for exogenous'):
        ARDL(data.y, 2, data.x, {'lry': [1, 2], 'ibo': 3, 'other': 4}, causal=False)
    with pytest.warns(SpecificationWarning, match='exog contains variables that'):
        ARDL(data.y, 2, data.x, {'lry': [1, 2]}, causal=False)