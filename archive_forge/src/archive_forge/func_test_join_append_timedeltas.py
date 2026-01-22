from datetime import (
import re
import numpy as np
import pytest
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import (
def test_join_append_timedeltas(self, using_array_manager):
    d = DataFrame.from_dict({'d': [datetime(2013, 11, 5, 5, 56)], 't': [timedelta(0, 22500)]})
    df = DataFrame(columns=list('dt'))
    msg = 'The behavior of DataFrame concatenation with empty or all-NA entries'
    warn = FutureWarning
    if using_array_manager:
        warn = None
    with tm.assert_produces_warning(warn, match=msg):
        df = concat([df, d], ignore_index=True)
        result = concat([df, d], ignore_index=True)
    expected = DataFrame({'d': [datetime(2013, 11, 5, 5, 56), datetime(2013, 11, 5, 5, 56)], 't': [timedelta(0, 22500), timedelta(0, 22500)]})
    if using_array_manager:
        expected = expected.astype(object)
    tm.assert_frame_equal(result, expected)