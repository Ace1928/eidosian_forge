import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_grouped_hist_multiple_axes_error(self, hist_df):
    df = hist_df
    fig, axes = mpl.pyplot.subplots(2, 3)
    msg = 'The number of passed axes must be 1, the same as the output plot'
    with pytest.raises(ValueError, match=msg):
        axes = df.hist(column='height', ax=axes)