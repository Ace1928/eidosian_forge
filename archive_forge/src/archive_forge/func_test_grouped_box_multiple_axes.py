import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_multiple_axes(self, hist_df):
    df = hist_df
    with tm.assert_produces_warning(UserWarning):
        _, axes = mpl.pyplot.subplots(2, 2)
        df.groupby('category').boxplot(column='height', return_type='axes', ax=axes)
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(2, 2))