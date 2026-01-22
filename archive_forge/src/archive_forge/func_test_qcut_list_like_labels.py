import os
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
from pandas.tseries.offsets import Day
@pytest.mark.parametrize('labels, expected', [(['a', 'b', 'c'], Categorical(['a', 'b', 'c'], ordered=True)), (list(range(3)), Categorical([0, 1, 2], ordered=True))])
def test_qcut_list_like_labels(labels, expected):
    values = range(3)
    result = qcut(values, 3, labels=labels)
    tm.assert_categorical_equal(result, expected)