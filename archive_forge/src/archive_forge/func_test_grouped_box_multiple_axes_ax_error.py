import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_multiple_axes_ax_error(self, hist_df):
    df = hist_df
    msg = 'The number of passed axes must be 3, the same as the output plot'
    with pytest.raises(ValueError, match=msg):
        fig, axes = mpl.pyplot.subplots(2, 3)
        with tm.assert_produces_warning(UserWarning):
            axes = df.groupby('classroom').boxplot(ax=axes)