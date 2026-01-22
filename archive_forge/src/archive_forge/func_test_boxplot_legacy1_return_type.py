import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_boxplot_legacy1_return_type(self, hist_df):
    grouped = hist_df.groupby(by='gender')
    axes = _check_plot_works(grouped.boxplot, subplots=False, return_type='axes')
    _check_axes_shape(axes, axes_num=1, layout=(1, 1))