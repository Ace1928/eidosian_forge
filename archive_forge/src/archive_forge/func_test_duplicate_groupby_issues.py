import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_duplicate_groupby_issues(self):
    idx_tp = [('600809', '20061231'), ('600809', '20070331'), ('600809', '20070630'), ('600809', '20070331')]
    dt = ['demo', 'demo', 'demo', 'demo']
    idx = MultiIndex.from_tuples(idx_tp, names=['STK_ID', 'RPT_Date'])
    s = Series(dt, index=idx)
    result = s.groupby(s.index).first()
    assert len(result) == 3