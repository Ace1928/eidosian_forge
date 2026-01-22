import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multilevel_consolidate(self):
    index = MultiIndex.from_tuples([('foo', 'one'), ('foo', 'two'), ('bar', 'one'), ('bar', 'two')])
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 4)), index=index, columns=index)
    df['Totals', ''] = df.sum(1)
    df = df._consolidate()