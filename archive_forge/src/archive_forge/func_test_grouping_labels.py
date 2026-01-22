from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_grouping_labels(self, multiindex_dataframe_random_data):
    grouped = multiindex_dataframe_random_data.groupby(multiindex_dataframe_random_data.index.get_level_values(0))
    exp_labels = np.array([2, 2, 2, 0, 0, 1, 1, 3, 3, 3], dtype=np.intp)
    tm.assert_almost_equal(grouped._grouper.codes[0], exp_labels)