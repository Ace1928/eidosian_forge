import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_boxplot_return_type_none(self, hist_df):
    result = hist_df.boxplot()
    assert isinstance(result, mpl.pyplot.Axes)