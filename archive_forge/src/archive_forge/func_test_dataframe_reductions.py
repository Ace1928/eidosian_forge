import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
@pytest.mark.parametrize('op', ['sum', 'prod', 'min', 'max'])
def test_dataframe_reductions(op):
    df = pd.DataFrame({'a': pd.array([1, 2], dtype='Int64')})
    result = df.max()
    assert isinstance(result['a'], np.int64)