from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('ops', [[lambda x: x + 1], [lambda x: x.sum()], ['sum', np.sum, lambda x: x.sum()], [lambda x: x + 1, lambda x: 3]])
def test_listlike_lambda_raises(ops):
    df = DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError, match='by_row=True not allowed'):
        df.apply(ops, by_row=True)