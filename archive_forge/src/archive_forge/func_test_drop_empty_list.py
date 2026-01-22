import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [[1, 2, 3], [1, 1, 2]])
@pytest.mark.parametrize('drop_labels', [[], [1], [2]])
def test_drop_empty_list(self, index, drop_labels):
    expected_index = [i for i in index if i not in drop_labels]
    frame = DataFrame(index=index).drop(drop_labels)
    tm.assert_frame_equal(frame, DataFrame(index=expected_index))