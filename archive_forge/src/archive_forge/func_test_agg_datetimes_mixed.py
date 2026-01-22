import datetime as dt
from functools import partial
import numpy as np
import pytest
from pandas.errors import SpecificationError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.io.formats.printing import pprint_thing
def test_agg_datetimes_mixed():
    data = [[1, '2012-01-01', 1.0], [2, '2012-01-02', 2.0], [3, None, 3.0]]
    df1 = DataFrame({'key': [x[0] for x in data], 'date': [x[1] for x in data], 'value': [x[2] for x in data]})
    data = [[row[0], dt.datetime.strptime(row[1], '%Y-%m-%d').date() if row[1] else None, row[2]] for row in data]
    df2 = DataFrame({'key': [x[0] for x in data], 'date': [x[1] for x in data], 'value': [x[2] for x in data]})
    df1['weights'] = df1['value'] / df1['value'].sum()
    gb1 = df1.groupby('date').aggregate('sum')
    df2['weights'] = df1['value'] / df1['value'].sum()
    gb2 = df2.groupby('date').aggregate('sum')
    assert len(gb1) == len(gb2)