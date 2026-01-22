from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('ops', [{'a': lambda x: x + 1}, {'a': lambda x: x.sum()}, {'a': ['sum', np.sum, lambda x: x.sum()]}, {'a': lambda x: 1}])
def test_dictlike_lambda_raises(ops):
    df = DataFrame({'a': [1, 2]})
    with pytest.raises(ValueError, match='by_row=True not allowed'):
        df.apply(ops, by_row=True)