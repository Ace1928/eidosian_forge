import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
@pytest.mark.parametrize('stat', SPECIAL, ids=[s[0] for s in SPECIAL])
def test_special_stats(df, stat):
    all_stats = [st for st in Description.default_statistics]
    all_stats.remove(stat[0])
    res = Description(df, stats=all_stats)
    for val in stat[1]:
        assert val not in res.frame.index