from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_series_yield_constant(df):
    result = df.groupby(['A', 'B'])['C'].apply(len)
    assert result.index.names[:2] == ('A', 'B')