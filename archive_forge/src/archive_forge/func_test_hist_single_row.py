import numpy as np
import pytest
from pandas import (
from pandas.tests.plotting.common import (
def test_hist_single_row(self):
    bins = np.arange(80, 100 + 2, 1)
    df = DataFrame({'Name': ['AAA', 'BBB'], 'ByCol': [1, 2], 'Mark': [85, 89]})
    df['Mark'].hist(by=df['ByCol'], bins=bins)