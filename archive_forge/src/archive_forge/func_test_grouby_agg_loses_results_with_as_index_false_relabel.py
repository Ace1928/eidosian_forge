import datetime
import functools
from functools import partial
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouby_agg_loses_results_with_as_index_false_relabel():
    df = DataFrame({'key': ['x', 'y', 'z', 'x', 'y', 'z'], 'val': [1.0, 0.8, 2.0, 3.0, 3.6, 0.75]})
    grouped = df.groupby('key', as_index=False)
    result = grouped.agg(min_val=pd.NamedAgg(column='val', aggfunc='min'))
    expected = DataFrame({'key': ['x', 'y', 'z'], 'min_val': [1.0, 0.8, 0.75]})
    tm.assert_frame_equal(result, expected)