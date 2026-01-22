import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_groupby_boxplot_object(self, hist_df):
    df = hist_df.astype('object')
    grouped = df.groupby('gender')
    msg = 'boxplot method requires numerical columns, nothing to plot'
    with pytest.raises(ValueError, match=msg):
        _check_plot_works(grouped.boxplot, subplots=False)