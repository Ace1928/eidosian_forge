from datetime import (
from itertools import product
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.core.reshape import reshape as reshape_lib
from pandas.core.reshape.pivot import pivot_table
def test_no_col(self, data):
    data.columns = [k * 2 for k in data.columns]
    msg = re.escape('agg function failed [how->mean,dtype->')
    with pytest.raises(TypeError, match=msg):
        data.pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
    table = data.drop(columns='CC').pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
    for value_col in table.columns:
        totals = table.loc[('All', ''), value_col]
        assert totals == data[value_col].mean()
    with pytest.raises(TypeError, match=msg):
        data.pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
    table = data.drop(columns='CC').pivot_table(index=['AA', 'BB'], margins=True, aggfunc='mean')
    for item in ['DD', 'EE', 'FF']:
        totals = table.loc[('All', ''), item]
        assert totals == data[item].mean()