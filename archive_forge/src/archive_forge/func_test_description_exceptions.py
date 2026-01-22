import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_description_exceptions():
    df = pd.DataFrame({'a': np.empty(100), 'b': pd.Series(np.arange(100) % 10)}, dtype='category')
    with pytest.raises(ValueError):
        Description(df, stats=['unknown'])
    with pytest.raises(ValueError):
        Description(df, alpha=-0.3)
    with pytest.raises(ValueError):
        Description(df, percentiles=[0, 100])
    with pytest.raises(ValueError):
        Description(df, percentiles=[10, 20, 30, 10])
    with pytest.raises(ValueError):
        Description(df, ntop=-3)
    with pytest.raises(ValueError):
        Description(df, numeric=False, categorical=False)