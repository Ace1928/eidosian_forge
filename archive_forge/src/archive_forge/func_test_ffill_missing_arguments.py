import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_ffill_missing_arguments():
    df = DataFrame({'a': [1, 2], 'b': [1, 1]})
    msg = 'DataFrameGroupBy.fillna is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pytest.raises(ValueError, match='Must specify a fill'):
            df.groupby('b').fillna()