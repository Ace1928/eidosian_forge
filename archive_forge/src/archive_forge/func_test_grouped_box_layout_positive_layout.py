import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_layout_positive_layout(self, hist_df):
    df = hist_df
    msg = 'At least one dimension of layout must be positive'
    with pytest.raises(ValueError, match=msg):
        df.boxplot(column=['weight', 'height'], by=df.gender, layout=(-1, -1))