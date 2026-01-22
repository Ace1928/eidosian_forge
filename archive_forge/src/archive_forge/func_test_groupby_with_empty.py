from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_groupby_with_empty(self):
    index = pd.DatetimeIndex(())
    data = ()
    series = Series(data, index, dtype=object)
    grouper = Grouper(freq='D')
    grouped = series.groupby(grouper)
    assert next(iter(grouped), None) is None