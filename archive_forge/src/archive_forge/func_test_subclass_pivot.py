import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_subclass_pivot(self):
    df = tm.SubclassedDataFrame({'index': ['A', 'B', 'C', 'C', 'B', 'A'], 'columns': ['One', 'One', 'One', 'Two', 'Two', 'Two'], 'values': [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]})
    pivoted = df.pivot(index='index', columns='columns', values='values')
    expected = tm.SubclassedDataFrame({'One': {'A': 1.0, 'B': 2.0, 'C': 3.0}, 'Two': {'A': 1.0, 'B': 2.0, 'C': 3.0}})
    expected.index.name, expected.columns.name = ('index', 'columns')
    tm.assert_frame_equal(pivoted, expected)