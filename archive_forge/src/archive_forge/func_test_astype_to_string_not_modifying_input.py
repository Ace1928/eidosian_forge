import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [None, 1, 1.5, np.nan, NaT])
def test_astype_to_string_not_modifying_input(string_storage, val):
    df = DataFrame({'a': ['a', 'b', val]})
    expected = df.copy()
    with option_context('mode.string_storage', string_storage):
        df.astype('string', copy=False)
    tm.assert_frame_equal(df, expected)