from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
def test_groupby_level_mapper(multiindex_dataframe_random_data):
    deleveled = multiindex_dataframe_random_data.reset_index()
    mapper0 = {'foo': 0, 'bar': 0, 'baz': 1, 'qux': 1}
    mapper1 = {'one': 0, 'two': 0, 'three': 1}
    result0 = multiindex_dataframe_random_data.groupby(mapper0, level=0).sum()
    result1 = multiindex_dataframe_random_data.groupby(mapper1, level=1).sum()
    mapped_level0 = np.array([mapper0.get(x) for x in deleveled['first']], dtype=np.int64)
    mapped_level1 = np.array([mapper1.get(x) for x in deleveled['second']], dtype=np.int64)
    expected0 = multiindex_dataframe_random_data.groupby(mapped_level0).sum()
    expected1 = multiindex_dataframe_random_data.groupby(mapped_level1).sum()
    expected0.index.name, expected1.index.name = ('first', 'second')
    tm.assert_frame_equal(result0, expected0)
    tm.assert_frame_equal(result1, expected1)