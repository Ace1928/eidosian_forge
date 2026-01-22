from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_raw_function_runs_once(engine):
    if engine == 'numba':
        pytest.skip('appending to list outside of numba func is not supported')
    df = DataFrame({'a': [1, 2, 3]})
    values = []

    def reducing_function(row):
        values.extend(row)

    def non_reducing_function(row):
        values.extend(row)
        return row
    for func in [reducing_function, non_reducing_function]:
        del values[:]
        df.apply(func, engine=engine, raw=True, axis=1)
        assert values == list(df.a.to_list())