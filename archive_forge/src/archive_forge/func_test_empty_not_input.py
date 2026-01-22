import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_empty_not_input(self):
    df = DataFrame(index=pd.DatetimeIndex([]))
    with tm.assert_produces_warning(FutureWarning, match=last_deprecated_msg):
        result = df.last(offset=1)
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = df.first(offset=1)
    tm.assert_frame_equal(df, result)
    assert df is not result