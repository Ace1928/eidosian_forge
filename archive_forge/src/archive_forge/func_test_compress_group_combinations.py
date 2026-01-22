import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_compress_group_combinations(self):
    key1 = [str(i) for i in range(10000)]
    key1 = np.tile(key1, 2)
    key2 = key1[::-1]
    df = DataFrame({'key1': key1, 'key2': key2, 'value1': np.random.default_rng(2).standard_normal(20000)})
    df2 = DataFrame({'key1': key1[::2], 'key2': key2[::2], 'value2': np.random.default_rng(2).standard_normal(10000)})
    merge(df, df2, how='outer')