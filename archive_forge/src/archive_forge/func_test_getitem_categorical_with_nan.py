import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_categorical_with_nan(self):
    ci = CategoricalIndex(['A', 'B', np.nan])
    ser = Series(range(3), index=ci)
    assert ser[np.nan] == 2
    assert ser.loc[np.nan] == 2
    df = DataFrame(ser)
    assert df.loc[np.nan, 0] == 2
    assert df.loc[np.nan][0] == 2