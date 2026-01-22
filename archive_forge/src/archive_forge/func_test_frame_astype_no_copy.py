import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_frame_astype_no_copy():
    df = DataFrame({'a': [1, 4, None, 5], 'b': [6, 7, 8, 9]}, dtype=object)
    result = df.astype({'a': Int16DtypeNoCopy()}, copy=False)
    assert result.a.dtype == pd.Int16Dtype()
    assert np.shares_memory(df.b.values, result.b.values)