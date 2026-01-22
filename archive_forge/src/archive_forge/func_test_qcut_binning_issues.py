import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
def test_qcut_binning_issues(datapath):
    cut_file = datapath(os.path.join('reshape', 'data', 'cut_data.csv'))
    arr = np.loadtxt(cut_file)
    result = qcut(arr, 20)
    starts = []
    ends = []
    for lev in np.unique(result):
        s = lev.left
        e = lev.right
        assert s != e
        starts.append(float(s))
        ends.append(float(e))
    for (sp, sn), (ep, en) in zip(zip(starts[:-1], starts[1:]), zip(ends[:-1], ends[1:])):
        assert sp < sn
        assert ep < en
        assert ep <= sn