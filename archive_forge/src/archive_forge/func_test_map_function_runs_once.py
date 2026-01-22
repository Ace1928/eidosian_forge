from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_map_function_runs_once():
    df = DataFrame({'a': [1, 2, 3]})
    values = []

    def reducing_function(val):
        values.append(val)

    def non_reducing_function(val):
        values.append(val)
        return val
    for func in [reducing_function, non_reducing_function]:
        del values[:]
        df.map(func)
        assert values == df.a.to_list()