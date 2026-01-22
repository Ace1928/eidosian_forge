import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_std_err(df):
    """
    Test the standard error of the mean matches result from scipy.stats.sem
    """
    np.testing.assert_allclose(Description(df['a']).frame.loc['std_err'], scipy.stats.sem(df['a']))