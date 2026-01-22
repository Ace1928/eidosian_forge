import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
def test_grouped_box_layout_needs_by(self, hist_df):
    df = hist_df
    msg = "The 'layout' keyword is not supported when 'by' is None"
    with pytest.raises(ValueError, match=msg):
        df.boxplot(column=['height', 'weight', 'category'], layout=(2, 1), return_type='dict')