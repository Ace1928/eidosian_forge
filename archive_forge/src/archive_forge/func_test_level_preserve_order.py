from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
@pytest.mark.parametrize('sort,labels', [[True, [2, 2, 2, 0, 0, 1, 1, 3, 3, 3]], [False, [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]]])
def test_level_preserve_order(self, sort, labels, multiindex_dataframe_random_data):
    grouped = multiindex_dataframe_random_data.groupby(level=0, sort=sort)
    exp_labels = np.array(labels, np.intp)
    tm.assert_almost_equal(grouped._grouper.codes[0], exp_labels)